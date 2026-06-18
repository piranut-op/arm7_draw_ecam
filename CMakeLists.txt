cmake_minimum_required(VERSION 3.8)
project(arm7_draw_ecam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Resource directories (no images/ — that was GUI-only)
install(
  DIRECTORY
    urdf
    launch
    config
    rviz
    meshes
  DESTINATION share/${PROJECT_NAME}/
)

# Python module (arm7_draw_ecam package)
ament_python_install_package(${PROJECT_NAME})

# The Python node executables transitively used by ecam_simple.launch.py
install(PROGRAMS
  ${PROJECT_NAME}/fk_arm_final.py
  ${PROJECT_NAME}/ik_arm_final.py
  ${PROJECT_NAME}/ecam_path.py
  ${PROJECT_NAME}/ecam_traj.py
  ${PROJECT_NAME}/ee_trail_marker.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
