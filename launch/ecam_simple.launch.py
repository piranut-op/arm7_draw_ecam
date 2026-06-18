#!/usr/bin/env python3
"""Draw-ECAM via the SIMPLE pipeline — one node, ONE trajectory, the controller.

  ros2 launch arm7_draw_ecam ecam_simple.launch.py                 # draw "ECAM" in Gazebo
  ros2 launch arm7_draw_ecam ecam_simple.launch.py plane:=ground   # pen DOWN, on the X-Y plane
  ros2 launch arm7_draw_ecam ecam_simple.launch.py shape:=ecam lin_vel:=0.01 rviz:=false

ecam_traj.py computes EVERYTHING offline — the letter path, a warm-started IK per
waypoint (the thesis DLS solver), the prepose and the return-home — and sends ONE
multi-point FollowJointTrajectory action goal. Multi-point trajectories are the
JointTrajectoryController's native input, so there are no streaming rules
(rates/velocities/horizons), no posture gate / prepose discovery race (delivery is
acknowledged), and nothing keeps publishing afterwards (no stale-commander bugs).

The plant is Ignition Gazebo (gazebo.launch.py + spawners → gz_ros2_control,
velocity JTC). fk_arm_final + the rosbag exist only for measurement.

Analyse the recorded bag afterwards (path printed below):
  python3 tools/verify_ecam_drawing.py --bag <bag_path> --plane yz --plot ~/ecam_plots
"""
import math
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _setup(context, *_, **__):
    lin_vel = float(LaunchConfiguration('lin_vel').perform(context))
    dwell_s = float(LaunchConfiguration('dwell_s').perform(context))
    pen_lift = float(LaunchConfiguration('pen_lift').perform(context))
    j2_deg = float(LaunchConfiguration('posture_j2_deg').perform(context))
    j4_deg = float(LaunchConfiguration('posture_j4_deg').perform(context))
    plane = LaunchConfiguration('plane').perform(context).lower()
    if plane not in ('vertical', 'ground'):
        raise RuntimeError(f"plane:={plane} — expected vertical | ground")
    position_only = LaunchConfiguration('position_only')
    shape = LaunchConfiguration('shape')
    rviz = LaunchConfiguration('rviz')

    # marker TIP (pen_tip, past the EE flange) except for the free-space helix,
    # which coils about the flange itself
    tip = 'ee' if shape.perform(context) == 'helix' else 'pen_tip'

    pkg = FindPackageShare('arm7_draw_ecam')

    # ── the plant: Ignition Gazebo + ros2_control spawners ───────────────────
    base = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, 'launch', 'gazebo.launch.py']))),
        Node(package='controller_manager', executable='spawner',
             arguments=['joint_state_broadcaster',
                        '--controller-manager', '/controller_manager'],
             output='screen'),
        Node(package='controller_manager', executable='spawner',
             arguments=['arm_controller',
                        '--controller-manager', '/controller_manager'],
             output='screen'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg, 'launch', 'rviz.launch.py'])),
            condition=IfCondition(rviz)),
    ]
    default_delay = 12.0          # Gazebo + spawners need a while

    delay_arg = LaunchConfiguration('delay').perform(context)
    delay = float(delay_arg) if delay_arg else default_delay

    # ── the command side: ecam_traj only (+ fk/trail/bag for measurement) ────
    fk = Node(package='arm7_draw_ecam', executable='fk_arm_final.py',
              name='fk_arm_final', output='screen',
              parameters=[{'tip_link': tip}])
    # ground writing uses the verified pen-down posture (derived by
    # tools/find_ground_posture.py); vertical uses the j2/j4 family.
    if plane == 'ground':
        from arm7_draw_ecam.ecam_path import GROUND_POSTURE
        posture = list(GROUND_POSTURE)
    else:
        posture = [0.0, math.radians(j2_deg), 0.0, math.radians(j4_deg),
                   0.0, 0.0, 0.0]
    traj = Node(package='arm7_draw_ecam', executable='ecam_traj.py',
                name='ecam_traj', output='screen',
                parameters=[{'tip_link': tip, 'shape': shape,
                             'position_only': position_only,
                             'posture_joints': posture,
                             'plane': plane,
                             'center_y': True, 'mirror_y': False,
                             'pen_lift': pen_lift,
                             'dwell_s': dwell_s,
                             'max_lin_vel': lin_vel,
                             'ideal_csv': '/tmp/ecam_ideal.csv'}])
    trail = Node(package='arm7_draw_ecam', executable='ee_trail_marker.py',
                 name='ee_trail_marker', output='screen',
                 parameters=[{'show_target_trail': False}])

    # auto-suffixed rosbag for verify_ecam_drawing.py
    bag_path = LaunchConfiguration('bag_path').perform(context) or '/tmp/ecam_simple_gazebo'
    unique_bag = bag_path
    i = 1
    while os.path.exists(unique_bag):
        unique_bag = f'{bag_path}_{i}'
        i += 1
    if plane == 'ground':
        print('[ecam_simple] gazebo  plane=ground  posture=GROUND_POSTURE (pen down)')
    else:
        print(f'[ecam_simple] gazebo  plane=vertical  '
              f'posture j2={j2_deg:.0f} deg j4={j4_deg:.0f} deg')
    proj = 'xy' if plane == 'ground' else 'yz'
    print(f'[ecam_simple] recording drawing to bag: {unique_bag}')
    print(f'[ecam_simple] analyse with:\n'
          f'    python3 tools/verify_ecam_drawing.py '
          f'--bag {unique_bag} --plane {proj} --plot ~/ecam_plots')
    bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', unique_bag,
             '/ee_target', '/ee_pose', '/ecam_pen'],
        output='log',
        condition=IfCondition(LaunchConfiguration('with_bag')))

    # ecam_traj waits internally for /robot_description, /joint_states and the
    # action server — the delays below just avoid startup log noise.
    return base + [
        TimerAction(period=delay, actions=[fk, trail]),
        TimerAction(period=delay + 0.5, actions=[bag]),
        TimerAction(period=delay + 2.0, actions=[traj]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('shape', default_value='ecam',
                              description='ecam | circle | square | helix | logo'),
        DeclareLaunchArgument('lin_vel', default_value='0.01',
                              description='drawing linear speed (m/s)'),
        DeclareLaunchArgument('dwell_s', default_value='0.2',
                              description='hold at stroke corners (s)'),
        DeclareLaunchArgument('pen_lift', default_value='-0.02',
                              description='pen retract; negative = lift in +X '
                                          '(pen points -X in the writing posture)'),
        DeclareLaunchArgument('position_only', default_value='false'),
        DeclareLaunchArgument('posture_j2_deg', default_value='-15.0',
                              description='shoulder lean (deg); keep >= -30 and '
                                          '<= -10 (word unreachable past -10)'),
        DeclareLaunchArgument('posture_j4_deg', default_value='75.0',
                              description='elbow bend (deg); j4 = j2 + 90 keeps '
                                          'the pen horizontal'),
        DeclareLaunchArgument('plane', default_value='vertical',
                              description='vertical (front, validated; pen -X, '
                                          'Y-Z plane) | ground (pen DOWN -Z, flat '
                                          'on the X-Y plane; uses GROUND_POSTURE, '
                                          'ignores posture_j2/j4_deg)'),
        DeclareLaunchArgument('delay', default_value='',
                              description='seconds before starting the pipeline; '
                                          'empty = 12 (Gazebo + spawners need a while)'),
        DeclareLaunchArgument('with_bag', default_value='true'),
        DeclareLaunchArgument('bag_path', default_value='',
                              description='rosbag output; empty = /tmp/ecam_simple_gazebo; '
                                          'auto-suffixed if it exists'),
        OpaqueFunction(function=_setup),
    ])
