#!/usr/bin/env python3
"""ECAM drawing as ONE offline-computed trajectory — the SIMPLE pipeline.

The streaming demo (ecam_path -> ik_arm_final -> ik_to_trajectory -> JTC) is a
live task-level cascade; with closed_loop:=false its joint stream is already a
deterministic function of the letter geometry, so nothing actually needs to be
computed at runtime. This node does the whole thing offline and hands the
controller its native input:

  1. anchor = FK(writing posture)            (deterministic, no /ee_pose wait)
  2. build the Cartesian letter path          (build_word_samples — the SAME
                                               geometry code as ecam_path)
  3. IK every waypoint, warm-started from the previous solution (ik_attempt —
     the thesis DLS solver, no restarts -> no branch hopping)
  4. assemble ONE JointTrajectory: prepose (current q -> posture, min-jerk) +
     draw + return-home, with per-point velocities and timestamps
  5. send it as a single FollowJointTrajectory ACTION goal (delivery is
     acknowledged and completion reported — no discovery races, no posture
     gate, no streaming rules, nothing left running to go stale)

While the goal executes, the ideal pose stream is replayed on /ee_target +
/ecam_pen (time-aligned bookkeeping, NOT in the control path) and the ideal
letterform CSV / RViz word markers are produced, so verify_ecam_drawing.py and
ee_trail_marker work unchanged.

Publishes:  /ee_target, /ecam_pen, /ee_markers (latched ideal word)
Action:     /arm_controller/follow_joint_trajectory (FollowJointTrajectory)
"""

import math
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

from arm7_draw_ecam.fk_arm_final import ArmKinematics, rot_to_quat, quat_to_rot
from arm7_draw_ecam.ik_arm_final import ik_attempt
from arm7_draw_ecam.ecam_path import (JOINT_NAMES, WRITING_POSTURE, GROUND_POSTURE,
                               build_word_samples, word_marker_array,
                               write_ideal_csv)


def min_jerk(s):
    """Minimum-jerk time scaling on s in [0,1] (zero vel/acc at both ends)."""
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


# ── shared planning helpers (module-level: reused by ecam_gui.py) ─────────────
def joint_move(q0, q1, duration, rate):
    """Min-jerk joint-space interpolation q0 -> q1 on the 1/rate grid.
    Returns the list of waypoints EXCLUDING q0 (the plant is already there),
    INCLUDING q1."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    n = max(2, int(round(duration * rate)))
    return [(1.0 - min_jerk((i + 1) / n)) * q0 + min_jerk((i + 1) / n) * q1
            for i in range(n)]


def solve_cartesian_path(kin, q_seed, samples, quats, *, position_only=False,
                         log=None, err=None):
    """Sequential warm-started IK over the Cartesian samples. Returns the Nx n
    joint array, or None if any waypoint cannot be solved to tolerance — the
    caller must then NOT send anything toward real motors."""
    qs = np.empty((len(samples), kin.n))
    q = np.asarray(q_seed, float).copy()
    T = np.eye(4)
    report = max(1, len(samples) // 10)
    for i, (p, quat) in enumerate(zip(samples, quats)):
        T[:3, :3] = quat_to_rot(float(quat[0]), float(quat[1]),
                                float(quat[2]), float(quat[3]))
        T[:3, 3] = np.asarray(p, float)
        # warm start from the previous solution, NO restarts/two-stage:
        # adjacent targets are ~v/rate apart (~0.2 mm) so this converges in a
        # few DLS iterations and can't hop solution branches mid-stroke.
        q, pe, oe = ik_attempt(kin, q, T, position_only=position_only,
                               two_stage=False, iters=100)
        if pe > 1e-4 or (not position_only and oe > 1e-3):
            # one careful retry (full two-stage, more iterations) before failing
            q, pe, oe = ik_attempt(kin, q, T, position_only=position_only,
                                   two_stage=True, iters=300)
        if pe > 1e-4 or (not position_only and oe > 1e-3):
            if err:
                err(f'IK failed at waypoint {i}/{len(samples)} '
                    f'(pos_err={pe*1000:.2f} mm ori_err={oe:.4f} rad) — aborting, '
                    'no trajectory sent. Is the posture/word box reachable? '
                    '(see CLAUDE.md: posture_j2_deg must stay in [-30, -10])')
            return None
        qs[i] = q
        if log and i % report == 0:
            log(f'offline IK {i}/{len(samples)}...')
    return qs


def build_fjt_goal(qm, dt, vel_clamp=3.0):
    """One FollowJointTrajectory goal from an Nx7 joint array on a uniform dt
    grid: per-point velocities by central finite difference (clamped, zero at
    the ends; dwell holds repeat identical positions so their diff is ~0)."""
    qm = np.asarray(qm, float)
    vel = np.zeros_like(qm)
    if len(qm) > 2:
        vel[1:-1] = (qm[2:] - qm[:-2]) / (2.0 * dt)
    np.clip(vel, -vel_clamp, vel_clamp, out=vel)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(JOINT_NAMES)
    for i in range(len(qm)):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in qm[i]]
        pt.velocities = [float(v) for v in vel[i]]
        t = (i + 1) * dt
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
        goal.trajectory.points.append(pt)
    return goal


class EcamTraj(Node):
    def __init__(self):
        super().__init__('ecam_traj')
        # drawing parameters — same names/semantics as ecam_path
        self.declare_parameter('max_lin_vel', 0.02)    # m/s
        self.declare_parameter('publish_rate', 50.0)   # Hz — the waypoint time grid
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('letter_w', 0.025)
        self.declare_parameter('letter_h', 0.05)
        self.declare_parameter('gap', 0.012)
        self.declare_parameter('pen_depth', 0.02)
        self.declare_parameter('pen_lift', 0.02)
        self.declare_parameter('ideal_csv', '/tmp/ecam_ideal.csv')
        self.declare_parameter('settle_time', 5.0)     # prepose segment duration (s)
        self.declare_parameter('posture_joints', list(WRITING_POSTURE))
        self.declare_parameter('return_home', True)
        self.declare_parameter('home_joint_vel', 0.6)  # rad/s for the return segment
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'ee')
        self.declare_parameter('mirror_y', False)
        self.declare_parameter('center_y', False)
        # writing plane: 'vertical' (front, validated) draws in the world Y-Z
        # plane with the pen along -X; 'ground' draws flat on the world X-Y plane
        # with the pen pointing -Z (PLANE_ROT[plane] rotates the same geometry).
        self.declare_parameter('plane', 'vertical')
        self.declare_parameter('dwell_s', 0.0)
        self.declare_parameter('shape', 'ecam')
        self.declare_parameter('circle_radius', 0.04)
        self.declare_parameter('square_side', 0.06)
        self.declare_parameter('helix_radius', 0.03)
        self.declare_parameter('helix_turns', 2.0)
        self.declare_parameter('helix_pitch', 0.06)
        # solver / transport
        self.declare_parameter('position_only', False)
        self.declare_parameter('action_name', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('vel_clamp', 3.0)       # rad/s cap on point velocities
        # Shrink the solver's joint limits by this margin (rad) so the plan
        # NEVER rides a hard stop. Measured failure without it: the word box
        # solves with joint_6 clipped to its +0.262 limit; commanded into the
        # stop, Ignition physics WEDGES the joint there (stuck the whole rest
        # of the drawing, 0.55 rad desired-actual error, letters smeared one
        # pitch per travel). 10 mrad costs nothing (0 IK fails, worst 28 um).
        self.declare_parameter('limit_margin', 0.01)

        gp = lambda n: self.get_parameter(n).value     # noqa: E731
        self.v_lin = float(gp('max_lin_vel'))
        self.rate = float(gp('publish_rate'))
        self.frame = gp('frame_id')
        self.posture = np.array([float(v) for v in gp('posture_joints')], float)
        if len(self.posture) != 7:
            self.get_logger().warn('posture_joints needs 7 values — using default')
            self.posture = np.array(WRITING_POSTURE, float)
        self.plane = str(gp('plane')).lower()
        if self.plane not in ('vertical', 'ground'):
            self.get_logger().warn(f"unknown plane '{self.plane}', using 'vertical'")
            self.plane = 'vertical'
        # safety for a bare `ros2 run` with plane:=ground and no posture override:
        # the default vertical posture points the pen sideways, not down, so swap
        # in the verified ground posture (the launch files pass it explicitly).
        if self.plane == 'ground' and np.allclose(self.posture, np.array(WRITING_POSTURE)):
            self.posture = np.array(GROUND_POSTURE, float)
            self.get_logger().info('plane=ground: defaulting to GROUND_POSTURE (pen down)')
        self.settle_time = max(float(gp('settle_time')), 0.5)
        self.return_home = bool(gp('return_home'))
        self.home_joint_vel = float(gp('home_joint_vel'))
        self.position_only = bool(gp('position_only'))
        self.vel_clamp = float(gp('vel_clamp'))
        self.ideal_csv = gp('ideal_csv')
        self.shape = str(gp('shape')).lower()

        self._kin = None
        self._q_now = None
        self._goal_sent = False
        self._result_in = False
        # replay bookkeeping, filled by _plan_and_send
        self._replay = None      # list of (t_from_start, pos(3), quat(4), pen(bool))
        self._replay_t0 = None
        self._replay_i = 0
        self._t_exec = None      # controller progress from action feedback (s)

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self._on_urdf, latched)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self._target_pub = self.create_publisher(PoseStamped, '/ee_target', 10)
        self._pen_pub = self.create_publisher(Bool, '/ecam_pen', 10)
        word_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._word_pub = self.create_publisher(MarkerArray, '/ee_markers', word_qos)
        self._word_markers = None
        self._client = ActionClient(self, FollowJointTrajectory,
                                    gp('action_name'))

        self.create_timer(0.5, self._maybe_plan)             # waits for inputs once
        self.create_timer(1.0 / self.rate, self._replay_tick)
        self.create_timer(1.0, self._publish_word_markers)
        self.get_logger().info('ecam_traj: waiting for /robot_description, '
                               '/joint_states and the trajectory action server...')

    # ── inputs ────────────────────────────────────────────────────────────────
    def _on_urdf(self, msg: String):
        if self._kin is not None:
            return
        try:
            kin = ArmKinematics.from_urdf(
                msg.data, self.get_parameter('base_link').value,
                self.get_parameter('tip_link').value)
            margin = float(self.get_parameter('limit_margin').value)
            kin.q_min = kin.q_min + margin    # keep the plan off the hard stops
            kin.q_max = kin.q_max - margin
            self._kin = kin
        except Exception as e:                                # noqa: BLE001
            self.get_logger().error(f'could not build kinematics: {e}')

    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in JOINT_NAMES):
            self._q_now = np.array([msg.position[idx[n]] for n in JOINT_NAMES], float)

    # ── plan once, send once ──────────────────────────────────────────────────
    def _maybe_plan(self):
        if self._goal_sent or self._kin is None or self._q_now is None:
            return
        if not self._client.server_is_ready():
            return
        self._goal_sent = True          # one shot — even on failure we don't retry blind
        try:
            self._plan_and_send()
        except Exception as e:                                # noqa: BLE001
            self.get_logger().error(f'planning failed: {e}')
            raise

    def _plan_and_send(self):
        kin = self._kin
        dt = 1.0 / self.rate
        gp = lambda n: self.get_parameter(n).value     # noqa: E731

        # 1. anchor at FK(posture) — same pose the streaming demo anchors at
        #    after its prepose+gate, but known deterministically up front.
        T_anchor = kin.fk(self.posture)
        home_pos = T_anchor[:3, 3].copy()
        home_quat = rot_to_quat(T_anchor[:3, :3])

        # 2. Cartesian letter path — the SAME shared geometry as ecam_path
        log = self.get_logger().info
        warn = self.get_logger().warn
        samples, quats, pen, ideal_rows, _, _ = build_word_samples(
            home_pos, home_quat, shape=self.shape,
            letter_w=float(gp('letter_w')), letter_h=float(gp('letter_h')),
            gap=float(gp('gap')), pen_depth=float(gp('pen_depth')),
            pen_lift=float(gp('pen_lift')), v_lin=self.v_lin, rate=self.rate,
            dwell_s=float(gp('dwell_s')), mirror_y=bool(gp('mirror_y')),
            center_y=bool(gp('center_y')), circle_radius=float(gp('circle_radius')),
            square_side=float(gp('square_side')), helix_radius=float(gp('helix_radius')),
            helix_turns=float(gp('helix_turns')), helix_pitch=float(gp('helix_pitch')),
            plane=self.plane, log=log, warn=warn)
        if ideal_rows:
            self._word_markers = word_marker_array(ideal_rows, self.frame)
            self._publish_word_markers()
            write_ideal_csv(self.ideal_csv, ideal_rows, log=log, warn=warn)

        # 3. offline IK along the path (warm-started from the posture)
        q_draw = solve_cartesian_path(kin, self.posture, samples, quats,
                                      position_only=self.position_only,
                                      log=log, err=self.get_logger().error)
        if q_draw is None:
            return
        log(f'offline IK done: {len(q_draw)} waypoints, 0 failures')

        # 4. one trajectory: prepose + draw + return-home on a uniform dt grid
        q_list = joint_move(self._q_now, self.posture, self.settle_time, self.rate)
        i_draw0 = len(q_list)
        q_list.extend(q_draw)
        i_ret0 = len(q_list)
        if self.return_home:
            q_end = q_draw[-1]
            dur = max(float(np.max(np.abs(q_end))) / max(self.home_joint_vel, 1e-3), 0.5)
            q_list.extend(joint_move(q_end, np.zeros(kin.n), dur, self.rate))
        qm = np.asarray(q_list)

        goal = build_fjt_goal(qm, dt, vel_clamp=self.vel_clamp)
        total = len(qm) * dt
        log(f'ONE trajectory assembled: {len(qm)} points, {total:.1f}s '
            f'(prepose {i_draw0 * dt:.1f}s + draw {(i_ret0 - i_draw0) * dt:.1f}s '
            f'+ return {(len(qm) - i_ret0) * dt:.1f}s) — sending action goal')

        # 5. the /ee_target + /ecam_pen bookkeeping replay (draw + return only,
        #    matching the streaming demo: nothing is published during the prepose)
        replay = []
        for i in range(i_draw0, i_ret0):
            k = i - i_draw0
            replay.append(((i + 1) * dt, samples[k], quats[k], bool(pen[k])))
        for i in range(i_ret0, len(qm)):
            T = kin.fk(qm[i])
            replay.append(((i + 1) * dt, T[:3, 3].copy(),
                           rot_to_quat(T[:3, :3]), False))
        self._replay = replay

        fut = self._client.send_goal_async(goal,
                                           feedback_callback=self._on_feedback)
        fut.add_done_callback(self._on_goal_response)

    # ── action plumbing ───────────────────────────────────────────────────────
    def _on_goal_response(self, fut):
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('trajectory goal REJECTED by the controller')
            return
        self._replay_t0 = self.get_clock().now().nanoseconds * 1e-9
        self._replay_i = 0
        self.get_logger().info('trajectory goal accepted — executing')
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_feedback(self, msg):
        # The controller's OWN progress along the trajectory. This is the replay
        # clock: in Gazebo the JTC runs on SIM time while this node runs on wall
        # time (~4% skew measured -> tens of mm of apparent tracking error if
        # the replay free-runs on the wall clock).
        d = msg.feedback.desired.time_from_start
        self._t_exec = d.sec + d.nanosec * 1e-9

    def _on_result(self, fut):
        self._result_in = True
        self._t_exec = float('inf')   # flush any replay tail past the last feedback
        res = fut.result()
        code = res.result.error_code if res else -999
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('ECAM drawing complete (returned home).')
        else:
            self.get_logger().error(
                f'trajectory finished with error_code={code} '
                f'({res.result.error_string if res else "no result"})')

    # ── replay (bookkeeping stream for verify_ecam_drawing / RViz trails) ─────
    def _replay_tick(self):
        if self._replay is None or self._replay_t0 is None:
            return
        if self._replay_i >= len(self._replay):
            if not self._result_in:
                return
            self._replay = None       # done — nothing keeps streaming afterwards
            return
        # replay clock: the controller's reported progress (clock-regime agnostic);
        # wall-clock elapsed only as a fallback until the first feedback arrives
        if self._t_exec is not None:
            t = self._t_exec
        else:
            t = self.get_clock().now().nanoseconds * 1e-9 - self._replay_t0
        # publish every sample whose schedule time has passed (catch-up safe)
        while (self._replay_i < len(self._replay)
               and self._replay[self._replay_i][0] <= t):
            _, pos, quat, pen = self._replay[self._replay_i]
            self._replay_i += 1
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame
            msg.pose = Pose()
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])
            self._target_pub.publish(msg)
            self._pen_pub.publish(Bool(data=pen))

    def _publish_word_markers(self):
        if self._word_markers is None:
            return
        now = self.get_clock().now().to_msg()
        for m in self._word_markers.markers:
            m.header.stamp = now
        self._word_pub.publish(self._word_markers)


def main():
    rclpy.init()
    node = EcamTraj()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
