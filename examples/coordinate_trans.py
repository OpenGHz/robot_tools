from robot_tools.coordinate import CoordinateTools
import numpy as np

""" pose（position+orientation）坐标系转换（translation+rotation） """
robot_in_world = (np.zeros(3), np.array([np.pi / 2, np.pi / 2, 0]))
target_in_robot = (np.zeros(3), np.array([0, np.pi / 4, 0]))
target_in_world = CoordinateTools.to_world_coordinate(target_in_robot, robot_in_world)
print(target_in_world)

""" orientation坐标系转换（仅rotation） """
current_euler = [np.pi / 2, np.pi / 2, 0]
rela_euler = [0, np.pi / 4, 0]
target_euler = CoordinateTools.to_target_orientation(rela_euler, current_euler)
print(target_euler)

assert np.allclose(target_in_world[1], target_euler)

"""robot arm custom to raw"""
custom_pose = (np.array([0.1, -0.2, 0.3]), np.array([-0.7, 0.5, 0.7]))
raw_in_custom = (np.array([-0.1, -0.2, -0.3]), np.array([1.0, 1.5, 0]))
custom_in_raw = CoordinateTools.pose_reverse(*raw_in_custom)
print(CoordinateTools.custom_to_raw(custom_pose, raw_in_custom))
print(CoordinateTools.custom_to_raw(custom_pose, custom_in_raw=custom_in_raw))