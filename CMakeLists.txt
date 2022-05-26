cmake_minimum_required(VERSION 3.0.2)
project(map2gazebo)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()

execute_process(COMMAND python -m pip install trimesh==3.7.14)
