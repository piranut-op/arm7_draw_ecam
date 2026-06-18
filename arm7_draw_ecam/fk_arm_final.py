#!/usr/bin/env python3
"""Forward kinematics for the 7-DOF arm — the authoritative FK.

Kinematics are built directly from the URDF on /robot_description, so the FK and
geometric Jacobian match robot_state_publisher / RViz / Gazebo to ~1e-13 m for
whatever URDF is loaded (no DH idealisation, no hand-derived constants to go
stale). The geometric Jacobian and the math helpers follow the same formulation
as the MATLAB script robot_arm_7dof.mlx (verified by tools/verify_arm_kinematics.py).

This file owns the kinematics: ArmKinematics (fk + jacobian) plus the helpers.
The inverse-kinematics node (ik_arm_final.py) imports ArmKinematics from here and
adds the IK on top — it does not re-implement the forward kinematics.

Node behaviour:
  SUB  /robot_description  std_msgs/String      (latched URDF)
  SUB  /joint_states       sensor_msgs/JointState
  PUB  /ee_pose            geometry_msgs/PoseStamped   (FK of /joint_states)
"""

import math
import numpy as np


JOINT_NAMES = [f'joint_{i}' for i in range(1, 8)]
PI_2 = math.pi / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (same formulation as robot_arm_7dof.mlx)
# ─────────────────────────────────────────────────────────────────────────────
def rotm2axang_vec(R):
    """Rotation matrix -> 3-vector axis*angle (same as the MATLAB helper)."""
    cos_th = (np.trace(R) - 1.0) / 2.0
    cos_th = max(-1.0, min(1.0, cos_th))
    th = math.acos(cos_th)
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-6:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / math.sqrt(M[i, i])
        return th * axis
    axis = (1.0 / (2.0 * math.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * axis


def mdh_transform(a, alpha, d, theta):
    """Single-row Modified-DH transform (== MATLAB `homogenoeus`):
    Tx(a) Rx(alpha) Tz(d) Rz(theta)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct,     -st,     0.0,   a    ],
        [st*ca,  ct*ca,  -sa,   -sa*d ],
        [st*sa,  ct*sa,   ca,    ca*d ],
        [0.0,    0.0,     0.0,   1.0  ],
    ], dtype=float)


def matlab_mdh_table(l1=0.11515, l2=0.23000, l3=0.21212, l4=0.05000):
    """The 10-row Modified-DH table from robot_arm_7dof.mlx (used only to
    cross-check the math against MATLAB; the node itself uses the URDF)."""
    return [
        [0.0,  0.0,   l1,  0.0  ],   # 0  joint 1
        [0.0, -PI_2,  0.0, 0.0  ],   # 1  joint 2
        [0.0,  PI_2,  l2,  0.0  ],   # 2  joint 3
        [0.0,  PI_2,  0.0, 0.0  ],   # 3  joint 4
        [0.0, -PI_2,  l3,  0.0  ],   # 4  joint 5
        [0.0,  0.0,   0.0,-PI_2 ],   # 5  fixed 5'
        [0.0, -PI_2,  0.0, 0.0  ],   # 6  joint 6
        [0.0,  0.0,   0.0,-PI_2 ],   # 7  fixed 6'
        [0.0,  PI_2,  0.0, 0.0  ],   # 8  joint 7
        [l4,   0.0,   0.0, 0.0  ],   # 9  fixed tool
    ]


MDH_REV_ROWS = [0, 1, 2, 3, 4, 6, 8]   # 0-based; MATLAB revRows = [1 2 3 4 5 7 9]


def _rpy_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def _origin_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_R(*rpy)
    T[:3, 3] = xyz
    return T


def _axis_R4(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    T = np.eye(4)
    T[:3, :3] = np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])
    return T


def rot_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s, 0.25*s)
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s, (R[2, 1]-R[1, 2])/s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s, (R[0, 2]-R[2, 0])/s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        return ((R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s, (R[1, 0]-R[0, 1])/s)


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Forward kinematics + geometric Jacobian
# ─────────────────────────────────────────────────────────────────────────────
class ArmKinematics:
    """FK + geometric Jacobian for a serial chain.

    Build with `from_urdf` (production: zero error vs the live robot) or
    `from_mdh` (to reproduce the MATLAB DH model for cross-checking). The
    geometric Jacobian column k is [ z_k x (p_ee - p_k) ; z_k ] in the base
    frame, stacked [linear; angular] — the same convention as MATLAB
    jacobian_geom."""

    def __init__(self):
        self.mode = None
        self.n = 0
        self.joint_names = []
        self.q_min = None
        self.q_max = None
        self._dh = None
        self._rev = None
        self._joints = None

    @classmethod
    def from_mdh(cls, dh_rows=None, rev_rows=None, lengths=None):
        k = cls()
        k.mode = 'mdh'
        if dh_rows is None:
            dh_rows = matlab_mdh_table(*(lengths or ()))
        k._dh = [list(r) for r in dh_rows]
        k._rev = list(rev_rows if rev_rows is not None else MDH_REV_ROWS)
        k.n = len(k._rev)
        k.joint_names = JOINT_NAMES[:k.n]
        k.q_min = np.full(k.n, -math.pi)
        k.q_max = np.full(k.n, math.pi)
        return k

    @classmethod
    def from_urdf(cls, urdf_xml, base='base_link', tip='ee'):
        from urdf_parser_py.urdf import URDF
        robot = URDF.from_xml_string(urdf_xml)
        parent_of = {j.child: j for j in robot.joints}
        chain = []
        link = tip
        while link != base:
            if link not in parent_of:
                raise ValueError(
                    f"link '{link}' has no parent joint on the path to '{base}'")
            j = parent_of[link]
            chain.append(j)
            link = j.parent
        chain.reverse()

        k = cls()
        k.mode = 'urdf'
        k._joints = []
        k.joint_names = []
        qlo, qhi = [], []
        for j in chain:
            xyz = list(j.origin.xyz) if (j.origin and j.origin.xyz) else [0, 0, 0]
            rpy = list(j.origin.rpy) if (j.origin and j.origin.rpy) else [0, 0, 0]
            axis = np.array(j.axis if j.axis is not None else [0, 0, 1], float)
            rev = j.type in ('revolute', 'continuous')
            k._joints.append({'T': _origin_T(xyz, rpy), 'axis': axis, 'rev': rev})
            if rev:
                k.joint_names.append(j.name)
                if j.type == 'revolute' and j.limit is not None:
                    qlo.append(float(j.limit.lower)); qhi.append(float(j.limit.upper))
                else:
                    qlo.append(-math.pi); qhi.append(math.pi)
        k.n = len(k.joint_names)
        k.q_min = np.array(qlo); k.q_max = np.array(qhi)
        return k

    @property
    def q_mid(self):
        return 0.5 * (self.q_min + self.q_max)

    def _chain(self, q):
        """Return (T_ee, axes); axes[k] = (z_k, p_k) — joint k axis (unit, base)
        and a point on it."""
        if self.mode == 'mdh':
            dh = [list(r) for r in self._dh]
            for k, r in enumerate(self._rev):
                dh[r][3] += q[k]
            T = np.eye(4)
            cum = [T.copy()]
            for row in dh:
                T = T @ mdh_transform(*row)
                cum.append(T.copy())
            axes = [(cum[r + 1][:3, 2].copy(), cum[r + 1][:3, 3].copy())
                    for r in self._rev]
            return T, axes
        else:
            T = np.eye(4)
            axes = []
            qi = 0
            for j in self._joints:
                T = T @ j['T']
                if j['rev']:
                    z = T[:3, :3] @ j['axis']
                    z = z / np.linalg.norm(z)
                    axes.append((z.copy(), T[:3, 3].copy()))
                    T = T @ _axis_R4(j['axis'], q[qi])
                    qi += 1
            return T, axes

    def fk(self, q):
        """4x4 base->EE transform."""
        return self._chain(np.asarray(q, float))[0]

    def jacobian(self, q):
        """6xN geometric Jacobian (base frame, [linear; angular]) and the EE
        transform. Column k = [ z_k x (p_ee - p_k) ; z_k ]."""
        T_ee, axes = self._chain(np.asarray(q, float))
        p_ee = T_ee[:3, 3]
        J = np.zeros((6, self.n))
        for k, (z, p) in enumerate(axes):
            J[:3, k] = np.cross(z, p_ee - p)
            J[3:, k] = z
        return J, T_ee


# ─────────────────────────────────────────────────────────────────────────────
# FK node
# ─────────────────────────────────────────────────────────────────────────────
def _build_node_class():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped

    class FkNode(Node):
        def __init__(self):
            super().__init__('fk_arm_final')
            self.declare_parameter('base_link', 'base_link')
            self.declare_parameter('tip_link', 'ee')
            self._base = self.get_parameter('base_link').value
            self._tip = self.get_parameter('tip_link').value

            self.kin = None
            self.idx = None

            latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
            self.create_subscription(String, '/robot_description', self._cb_urdf, latched)
            self.create_subscription(JointState, '/joint_states', self._cb_js, 30)
            self.pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
            self.get_logger().info(
                f'fk_arm_final waiting for /robot_description ({self._base} -> {self._tip})')

        def _cb_urdf(self, msg):
            if self.kin is not None:
                return
            try:
                self.kin = ArmKinematics.from_urdf(msg.data, self._base, self._tip)
            except Exception as e:
                self.get_logger().error(f'URDF parse failed: {e}')
                return
            self.get_logger().info(
                f'URDF loaded: {self.kin.n} DoF — joints {self.kin.joint_names}')

        def _cb_js(self, msg):
            if self.kin is None:
                return
            if self.idx is None:
                try:
                    self.idx = [msg.name.index(n) for n in self.kin.joint_names]
                except ValueError:
                    return
            q = np.array([msg.position[i] for i in self.idx], float)
            T = self.kin.fk(q)
            qx, qy, qz, qw = rot_to_quat(T[:3, :3])
            m = PoseStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self._base
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (
                float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
            m.pose.orientation.x, m.pose.orientation.y = qx, qy
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            self.pub.publish(m)

    return FkNode


def main():
    import rclpy
    rclpy.init()
    node = _build_node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
