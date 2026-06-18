import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Package directories
    pkg_share = get_package_share_directory("arm7_draw_ecam")

    # ──────────────── GZ_SIM_RESOURCE_PATH (minimal – adjust as needed) ────────────────
    model_paths = [
        str(Path(pkg_share).parent.resolve()),  # often useful for local models
    ]
    # Add this line only if you actually need models from panda_description
    # model_paths.append(os.path.join(get_package_share_directory("panda_description"), "models"))

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pathsep.join(model_paths)
    )

    # ──────────────── Robot description (hardcoded xacro from your package) ────────────────
    xacro_path = os.path.join(pkg_share, "urdf", "arm7_draw_ecam.urdf.xacro")

    robot_description_content = Command([
        "xacro ", xacro_path,
        " is_ignition:=True"   # keep if your xacro uses this argument
    ])

    robot_description = ParameterValue(robot_description_content, value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }]
    )

    # ──────────────── Gazebo – default (no explicit world → uses Gazebo's default) ────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            )
        ]),
        # No gz_args → uses default (usually empty simulation)
        # If you want explicit empty world later, you can add:
        # launch_arguments={"gz_args": "-r -v 4 empty.sdf"}.items()
        launch_arguments={
            'gz_args': '-r -v 4 empty.sdf'   # ← this is the key line
            # If you want even more logging: '-r -v 6 empty.sdf'
        }.items()
    )

    # ──────────────── Spawn your robot ────────────────
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "arm7_draw_ecam",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.0",
            "-R", "0.0",
            "-P", "0.0",
            "-Y", "0.0",
        ]
    )

    # ──────────────── Parameter bridge ────────────────
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # Add more bridges here when needed (joint states, image_raw, etc.)
        ]
    )

    # Image bridge – uncomment when you need /camera/image_raw in ROS
    # ros_gz_image_bridge = Node(
    #     package="ros_gz_image",
    #     executable="image_bridge",
    #     arguments=["/camera/image_raw"],
    #     output="screen"
    # )

    return LaunchDescription([
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge,
        # ros_gz_image_bridge,
    ])