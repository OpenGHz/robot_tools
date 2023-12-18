import argparse

parser = argparse.ArgumentParser("Base control node config.")

parser.add_argument(
    "-cp",
    "--current_pose_topic",
    type=str,
    default="/airbot/pose",
    help="topic name of current pose, the type should be geometry_msgs/Pose",
)
parser.add_argument(
    "-tp",
    "--target_pose_topic",
    type=str,
    default="/airbot/base_pose_cmd",
    help="topic name of target pose, the type should be geometry_msgs/Pose",
)
parser.add_argument(
    "-vp",
    "--target_velocity_topic",
    type=str,
    default="/cmd_vel",
    help="topic name of velocity, the type should be geometry_msgs/Twist",
)

args, unknown = parser.parse_known_args()

current_pose_topic = args.current_pose_topic
target_pose_topic = args.target_pose_topic
target_velocity_topic = args.target_velocity_topic

from robot_tools import BaseControl

BaseControl._TEST_ = False
base_control = BaseControl()  # 初始化base位姿控制接口
base_control.set_move_kp(1.1, 1.1)
base_control.set_velocity_limits((0.2, 1.0), (0.3, 1.0))
base_control.set_direction_tolerance((0.002, 0.017 * 90))
base_control.set_wait_tolerance(0.01, 0.017 * 2, 60, 200)
base_control.avoid_321()

import numpy as np
from geometry_msgs.msg import Pose, Twist

base_pose_control_flag = False


def current_pose_sub(msg: Pose):
    pos = [msg.position.x, msg.position.y, msg.position.z]
    ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    base_control.set_current_world_pose(
        np.array(pos, dtype=np.float64), np.array(ori, dtype=np.float64)
    )
    # print("current pose received", pos, ori)


def target_pose_sub(msg: Pose):
    global base_pose_control_flag
    pos = [msg.position.x, msg.position.y, msg.position.z]
    ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    base_control.set_target_pose(
        np.array(pos, dtype=np.float64), np.array(ori, dtype=np.float64)
    )
    base_pose_control_flag = True
    # print("target pose received", pos, ori)


import rospy

node_name = "base_control_node"
rospy.init_node(node_name)
print(f"node {node_name} started:")
print(f"  -current_pose_topic: {current_pose_topic}")
print(f"  -target_pose_topic: {target_pose_topic}")
print(f"  -target_velocity_topic: {target_velocity_topic}")
current_pose_suber = rospy.Subscriber(
    current_pose_topic, Pose, current_pose_sub, queue_size=1
)
target_pose_suber = rospy.Subscriber(
    target_pose_topic, Pose, target_pose_sub, queue_size=1
)
puber = rospy.Publisher(target_velocity_topic, Twist, queue_size=1)
vel_msg = Twist()
rate = rospy.Rate(200)
while not rospy.is_shutdown():
    # base pose control
    if base_pose_control_flag:
        cmd = base_control.get_velocity_cmd(ignore_stop=True)
        vel_msg.linear.x = cmd[0][0]
        vel_msg.linear.y = cmd[0][1]
        vel_msg.angular.z = cmd[1][2]
        puber.publish(vel_msg)
        if (cmd[0] == 0).all() and (cmd[1] == 0).all():
            base_pose_control_flag = False
            print("base pose control finished")
    rate.sleep()
