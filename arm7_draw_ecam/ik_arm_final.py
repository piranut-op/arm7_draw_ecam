#!/usr/bin/env python3
"""Inverse kinematics for the 7-DOF arm — the authoritative IK.

The forward kinematics live in fk_arm_final.py; this file imports ArmKinematics
(FK + geometric Jacobian) from there and only adds the inverse solver — it does
not re-implement the forward kinematics.

The solver follows the same formulation as the MATLAB script robot_arm_7dof.mlx:
Damped Least Squares on the geometric Jacobian, with the 6-D Cartesian error
[position; axis-angle] and the orientation error in the base frame
(rotm2axang_vec(R_des * R_cur')). On top of that core it adds, for robustness:
adaptive damping, null-space joint-centring, joint-limit clipping, and — crucially
for arbitrary targets — RANDOM RESTARTS. A single DLS pass is a local method and
only reaches ~58% of reachable poses from a fixed seed; restarting from several
seeds (plus a position-first warm-up) reaches essentially all reachable poses.

Node behaviour:
  SUB  /robot_description  std_msgs/String      (latched URDF)
  SUB  /joint_states       sensor_msgs/JointState
  SUB  /ee_target          geometry_msgs/PoseStamped
  PUB  /joint_commands     sensor_msgs/JointState

On each new /ee_target the node solves once (warm-started from the current pose,
with restarts as needed) and then ramps the command toward the solution at
dq_max per tick so the motion is smooth. /ee_pose is published by fk_arm_final.
"""

import math
import numpy as np

from arm7_draw_ecam.fk_arm_final import (
    ArmKinematics, JOINT_NAMES, rotm2axang_vec, quat_to_rot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-kinematics math (uses ArmKinematics.jacobian from fk_arm_final)
# ─────────────────────────────────────────────────────────────────────────────
def ik_update(kin, q, T_des, *, position_only=False, lam=0.05, lam_min=5e-4,
              lam_knee=0.05, null_k=0.3, step_clamp=0.10):
    """One Damped-Least-Squares step. Returns (dq, e_pos, e_rot).

    Core (== MATLAB ik_dls): dq = J' (J J' + lam^2 I)^-1 e, with
    e = [p_des - p_cur ; rotm2axang_vec(R_des R_cur')] (base frame).
    Extras: adaptive damping (lam shrinks near the goal), null-space centring
    toward mid-limits, and a per-step ||dq|| cap."""
    J, T = kin.jacobian(q)
    e_pos = T_des[:3, 3] - T[:3, 3]
    e_rot = rotm2axang_vec(T_des[:3, :3] @ T[:3, :3].T)
    if position_only:
        Ju, e, m = J[:3, :], e_pos, 3
    else:
        Ju, e, m = J, np.concatenate([e_pos, e_rot]), 6

    lam_eff = lam_min + (lam - lam_min) * min(1.0, np.linalg.norm(e) / lam_knee)
    JT = Ju.T
    M = Ju @ JT + (lam_eff ** 2) * np.eye(m)
    dq = JT @ np.linalg.solve(M, e)

    if null_k:
        N = np.eye(kin.n) - JT @ np.linalg.solve(M, Ju)
        dq = dq + N @ (null_k * (kin.q_mid - q))

    nn = float(np.linalg.norm(dq))
    if nn > step_clamp:
        dq *= step_clamp / nn
    return dq, e_pos, e_rot


def _refine(kin, seed, T_des, *, iters, null_k, position_only,
            pos_tol, ori_tol, **kw):
    q = np.array(seed, float)
    ep = np.zeros(3); er = np.zeros(3)
    for _ in range(iters):
        dq, ep, er = ik_update(kin, q, T_des, null_k=null_k,
                               position_only=position_only, **kw)
        q = np.clip(q + dq, kin.q_min, kin.q_max)
        if np.linalg.norm(ep) < pos_tol and (position_only or np.linalg.norm(er) < ori_tol):
            break
    return q, float(np.linalg.norm(ep)), (0.0 if position_only else float(np.linalg.norm(er)))


def ik_attempt(kin, seed, T_des, *, position_only=False, two_stage=True,
               null_k=0.0, iters=300, pos_tol=1e-5, ori_tol=1e-4, **kw):
    """One attempt from `seed`. For full-pose targets a position-only warm-up
    (big basin of attraction, no null-space) lands near the target first, then a
    full 6-D refinement converges orientation — far more reliable than going
    straight to full pose."""
    if two_stage and not position_only:
        seed, _, _ = _refine(kin, seed, T_des, iters=max(80, iters // 2),
                             null_k=0.0, position_only=True,
                             pos_tol=pos_tol, ori_tol=ori_tol, **kw)
    return _refine(kin, seed, T_des, iters=iters, null_k=null_k,
                   position_only=position_only, pos_tol=pos_tol, ori_tol=ori_tol, **kw)


def ik_solve(kin, T_des, *, q0=None, restarts=8, iters=300, position_only=False,
             two_stage=True, null_k=0.0, pos_tol=1e-5, ori_tol=1e-4,
             seed_rng=None, **kw):
    """Robust IK with random restarts. Tries q0 (or rest) and the mid-limit
    pose first, then random seeds within the joint limits; returns the best
    (q, info). Stops early as soon as a seed converges."""
    rng = seed_rng if seed_rng is not None else np.random.default_rng()
    q_start = np.zeros(kin.n) if q0 is None else np.asarray(q0, float)
    fixed = [q_start, kin.q_mid]
    best_q = q_start; best_sc = np.inf; best = (np.inf, np.inf)
    for i in range(2 + max(0, restarts)):
        seed = fixed[i] if i < len(fixed) else rng.uniform(kin.q_min, kin.q_max)
        q, pe, oe = ik_attempt(kin, seed, T_des, position_only=position_only,
                               two_stage=two_stage, null_k=null_k, iters=iters,
                               pos_tol=pos_tol, ori_tol=ori_tol, **kw)
        sc = pe + (0.0 if position_only else oe)
        if sc < best_sc:
            best_sc, best_q, best = sc, q, (pe, oe)
        if pe < pos_tol and (position_only or oe < ori_tol):
            break
    return best_q, {'pos_err': best[0], 'ori_err': best[1],
                    'converged': best[0] < pos_tol and (position_only or best[1] < ori_tol)}


# ─────────────────────────────────────────────────────────────────────────────
# IK node
# ─────────────────────────────────────────────────────────────────────────────
def _build_node_class():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped

    class IkNode(Node):
        def __init__(self):
            super().__init__('ik_arm_final')
            self.declare_parameter('base_link', 'base_link')
            self.declare_parameter('tip_link', 'ee')
            self.declare_parameter('rate_hz', 200.0)
            self.declare_parameter('position_only', False)
            self.declare_parameter('closed_loop', True)
            self.declare_parameter('null_k', 0.0)   # 0 reaches far more targets; >0 centres posture
            self.declare_parameter('dq_max', 0.10)      # ramp cap, rad/tick
            self.declare_parameter('restarts', 8)
            self.declare_parameter('iters', 300)
            self.declare_parameter('pos_tol', 1e-5)
            self.declare_parameter('ori_tol', 1e-4)
            # Stop publishing /joint_commands after this many seconds without a
            # fresh /ee_target (0 = never, the legacy behaviour). The controller
            # holds the last trajectory when the stream stops, so normal motion
            # and the emergent "stop" are unchanged — but an ORPHANED ik node
            # left over from a killed run goes quiet instead of streaming its
            # stale solution forever and silently overriding the next run's
            # writing-posture trajectory (seen live: prepose replaced every
            # 40 ms, arm frozen off-posture, posture gate erroring).
            self.declare_parameter('target_timeout', 0.0)

            gp = lambda n: self.get_parameter(n).value
            self._base = gp('base_link'); self._tip = gp('tip_link')
            self.position_only = bool(gp('position_only'))
            self.closed_loop = bool(gp('closed_loop'))
            self.null_k = float(gp('null_k'))
            self.dq_max = float(gp('dq_max'))
            self.restarts = int(gp('restarts'))
            self.iters = int(gp('iters'))
            self.pos_tol = float(gp('pos_tol')); self.ori_tol = float(gp('ori_tol'))
            self.target_timeout = float(gp('target_timeout'))
            self._t_last_target = None

            self.kin = None
            self.idx = None
            self.q_fb = None
            self.q_cmd = None
            self.q_goal = None
            self.T_des = None
            self.need_solve = False
            self._rng = np.random.default_rng()

            latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
            self.create_subscription(String, '/robot_description', self._cb_urdf, latched)
            self.create_subscription(JointState, '/joint_states', self._cb_js, 30)
            self.create_subscription(PoseStamped, '/ee_target', self._cb_target, 10)
            self.pub = self.create_publisher(JointState, '/joint_commands', 20)
            self.create_timer(1.0 / float(gp('rate_hz')), self._tick)
            self.get_logger().info(
                f'ik_arm_final waiting for /robot_description ({self._base} -> {self._tip}); '
                f'restarts={self.restarts} closed_loop={self.closed_loop} '
                f'position_only={self.position_only}')

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
            self.q_fb = np.array([msg.position[i] for i in self.idx], float)

        def _cb_target(self, msg):
            o = msg.pose.orientation
            T = np.eye(4)
            T[:3, :3] = quat_to_rot(o.x, o.y, o.z, o.w)
            T[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self.T_des = T
            self.need_solve = True
            self._t_last_target = self.get_clock().now().nanoseconds * 1e-9

        def _tick(self):
            if self.kin is None or self.q_fb is None:
                return
            if self.q_cmd is None:
                self.q_cmd = self.q_fb.copy()

            if self.need_solve and self.T_des is not None:
                warm = self.q_fb if self.closed_loop else self.q_cmd
                self.q_goal, info = ik_solve(
                    self.kin, self.T_des, q0=warm, restarts=self.restarts,
                    iters=self.iters, position_only=self.position_only,
                    null_k=self.null_k, pos_tol=self.pos_tol, ori_tol=self.ori_tol,
                    seed_rng=self._rng)
                self.need_solve = False
                # NOTE: keep the info() and warn() calls on separate source lines.
                # rclpy caches a logger call by its caller location and forbids the
                # same location from switching severity ("Logger severity cannot be
                # changed between calls"), which a single lvl = info-or-warn call hits
                # the second time a target's converged-status differs.
                summary = (f"IK solve: pos_err={info['pos_err']*1000:.3f} mm "
                           f"ori_err={math.degrees(info['ori_err']):.3f} deg")
                if info['converged']:
                    self.get_logger().info(summary + ' converged')
                else:
                    self.get_logger().warn(
                        summary + ' BEST-EFFORT (target near/outside reach)')

            if self.q_goal is None:
                return
            if (self.target_timeout > 0.0 and self._t_last_target is not None
                    and (self.get_clock().now().nanoseconds * 1e-9
                         - self._t_last_target) > self.target_timeout):
                return  # target stream stopped — go quiet (controller holds)
            # ramp the command toward the solution for smooth motion
            step = np.clip(self.q_goal - self.q_cmd, -self.dq_max, self.dq_max)
            self.q_cmd = self.q_cmd + step

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = list(self.kin.joint_names)
            cmd.position = [float(v) for v in self.q_cmd]
            self.pub.publish(cmd)

    return IkNode


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
