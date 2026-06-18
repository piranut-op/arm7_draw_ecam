#!/usr/bin/env python3
"""Visualise the end-effector trail in RViz.

Subscribes:
  /ee_pose      — the actual EE pose (from fk_arm_dh_urdf or fk_arm_v3)
  /ee_target    — the IK target (optional; shown as a sphere if available)

Publishes:
  /ee_markers   — visualization_msgs/MarkerArray containing:
                  • LINE_STRIP "ee_trail"    — actual EE path
                  • LINE_STRIP "target_trail" — commanded target path
                  • SPHERE     "ee_now"      — current EE position
                  • SPHERE     "target"      — current /ee_target

Add a MarkerArray display in RViz subscribed to /ee_markers.

Parameters:
  trail_length     (2000)         — max points kept in the trail (older drop off)
  ee_color         (orange)       — RGBA for actual trail; comma-separated 0-1
  target_color     (cyan)         — RGBA for target trail
  ee_marker_size   (0.008)        — sphere diameter for current EE
  target_marker_size (0.012)      — sphere diameter for current target
  line_width       (0.003)        — strip width (m)
  reset_distance   (0.5)          — m; if a new pose is farther than this from
                                    the previous, clear the trail (e.g. user
                                    moved to a totally new pose)
  base_frame       ('base_link')  — frame_id used for all markers
  publish_rate     (20.0)         — Hz
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def parse_color(s, default):
    """Parse 'r,g,b,a' (0-1 floats) or one of: orange, cyan, red, green, blue, yellow, magenta, white."""
    NAMED = {
        'orange':  (1.0, 0.55, 0.0, 1.0),
        'cyan':    (0.0, 1.0,  1.0, 1.0),
        'red':     (1.0, 0.0,  0.0, 1.0),
        'green':   (0.0, 1.0,  0.0, 1.0),
        'blue':    (0.2, 0.4,  1.0, 1.0),
        'yellow':  (1.0, 1.0,  0.0, 1.0),
        'magenta': (1.0, 0.0,  1.0, 1.0),
        'white':   (1.0, 1.0,  1.0, 1.0),
    }
    if isinstance(s, str) and s.lower() in NAMED:
        return NAMED[s.lower()]
    if isinstance(s, str) and ',' in s:
        try:
            parts = [float(p) for p in s.split(',')]
            if len(parts) == 3:
                parts.append(1.0)
            if len(parts) == 4:
                return tuple(parts)
        except ValueError:
            pass
    return default


class EeTrailMarker(Node):
    def __init__(self):
        super().__init__('ee_trail_marker')
        self.declare_parameter('trail_length',       2000)
        self.declare_parameter('ee_color',           'orange')
        self.declare_parameter('target_color',       'cyan')
        self.declare_parameter('ee_marker_size',     0.008)
        self.declare_parameter('target_marker_size', 0.012)
        self.declare_parameter('line_width',         0.003)
        self.declare_parameter('reset_distance',     0.5)
        self.declare_parameter('base_frame',         'base_link')
        self.declare_parameter('publish_rate',       20.0)
        # Draw the commanded /ee_target trail (the growing cyan line + sphere). Set
        # False when something else already shows the ideal path as a static reference
        # (e.g. ecam_path publishes the full blue ECAM word on /ee_markers) — then the
        # only blue is that static word and orange (ee_trail) is the live EE.
        self.declare_parameter('show_target_trail',  True)

        gp = lambda n: self.get_parameter(n).value
        self.trail_len      = int(gp('trail_length'))
        self.color_ee       = parse_color(gp('ee_color'),     (1.0, 0.55, 0.0, 1.0))
        self.color_target   = parse_color(gp('target_color'), (0.0, 1.0, 1.0, 1.0))
        self.ee_size        = float(gp('ee_marker_size'))
        self.target_size    = float(gp('target_marker_size'))
        self.line_width     = float(gp('line_width'))
        self.reset_dist     = float(gp('reset_distance'))
        self.base_frame     = gp('base_frame')
        rate                = float(gp('publish_rate'))
        self.show_target    = bool(gp('show_target_trail'))

        self.ee_pts: list[tuple[float, float, float]] = []
        self.target_pts: list[tuple[float, float, float]] = []
        self.last_ee: np.ndarray | None = None
        self.last_target: np.ndarray | None = None

        self.create_subscription(PoseStamped, '/ee_pose',  self._on_ee,  10)
        self.create_subscription(PoseStamped, '/ee_target', self._on_tg, 10)
        self.pub = self.create_publisher(MarkerArray, '/ee_markers', 10)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'ee_trail_marker: publishing /ee_markers at {rate:.1f} Hz '
            f'(trail length {self.trail_len}, base {self.base_frame})')

    def _append(self, lst, p):
        lst.append((float(p[0]), float(p[1]), float(p[2])))
        if len(lst) > self.trail_len:
            del lst[0:len(lst) - self.trail_len]

    def _on_ee(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if (self.last_ee is not None
                and np.linalg.norm(p - self.last_ee) > self.reset_dist):
            self.ee_pts.clear()
        self._append(self.ee_pts, p)
        self.last_ee = p

    def _on_tg(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if (self.last_target is not None
                and np.linalg.norm(p - self.last_target) > self.reset_dist):
            self.target_pts.clear()
        self._append(self.target_pts, p)
        self.last_target = p

    def _make_strip(self, name, ns_id, pts, color, width):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = name
        m.id = ns_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = width
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        m.pose.orientation.w = 1.0
        m.points = [Point(x=x, y=y, z=z) for (x, y, z) in pts]
        return m

    def _make_sphere(self, name, ns_id, p, color, size):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = name
        m.id = ns_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = size
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
        m.pose.orientation.w = 1.0
        return m

    def _tick(self):
        msg = MarkerArray()
        if len(self.ee_pts) >= 2:
            msg.markers.append(self._make_strip(
                'ee_trail', 0, self.ee_pts, self.color_ee, self.line_width))
        if self.show_target and len(self.target_pts) >= 2:
            msg.markers.append(self._make_strip(
                'target_trail', 1, self.target_pts, self.color_target, self.line_width * 0.7))
        if self.last_ee is not None:
            msg.markers.append(self._make_sphere(
                'ee_now', 2, self.last_ee, self.color_ee, self.ee_size))
        if self.show_target and self.last_target is not None:
            msg.markers.append(self._make_sphere(
                'target', 3, self.last_target, self.color_target, self.target_size))
        if msg.markers:
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = EeTrailMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
