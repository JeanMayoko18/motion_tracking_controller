cmake_minimum_required(VERSION 3.5)
project(motion_tracking_controller)

# Indique que le package se construit via setup.py
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Installer les launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
