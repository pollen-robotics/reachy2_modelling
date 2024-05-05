import sys

import numpy as np
import roboticstoolbox as rtb

# general_robotics_toolbox.urdf.robot_from_xml_string
import sys
from pathlib import Path

urdf_path = str(Path(__file__).parent / "reachy.urdf")


def print_debug(*args):
    pass
    # print(*args)


def add_linknames(robot):
    robot.ee_links_names = [x.name for x in robot.ee_links]
    robot.links_names = [x.name for x in robot.links]


# TODO: links not working
def robot_from_urdf(filepath, link_names=None, gripper_name=None):
    links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(filepath, tld=".")

    if (
        gripper_name is not None
        and link_names is not None
        and gripper_name not in link_names
    ):
        print("gripper_name must be in link_names")
        print("gripper_name:", gripper_name)
        print("link_names:", link_names)
        sys.exit(1)

    filtered_links = links
    gripper_link = []
    leftout_links = []
    if link_names is not None:
        filtered_links = []
        for link in links:
            if link.name in link_names:
                filtered_links.append(link)
            else:
                leftout_links.append(link)
            if link.name == gripper_name:
                print_debug("link:", link.name, "gripper:", gripper_name)
                gripper_link.append(link)

        # print([x.name for x in filtered_links])

        if len(filtered_links) != len(link_names):
            print("filtered_links != link_names")
            sys.exit(1)
    print_debug("left out:", [x.name for x in leftout_links])
    print_debug("included:", [x.name for x in filtered_links])

    if len(gripper_link) > 1:
        print("gripper_link > 1")
        sys.exit(1)

    robot = rtb.Robot(
        filtered_links,
        gripper_links=gripper_link,
        urdf_string=urdf_string,
        urdf_filepath=urdf_filepath,
    )
    add_linknames(robot)
    return robot


# dhrobot = rtb.DHRobot.URDF_read(
#     urdf_path, tld='.')  #, root_link='torso', tip='r_arm_tip')
# print([type(x) for x in dhrobot])

# erobot = rtb.ERobot.URDF_read(urdf_path, tld='.')
# print([type(x) for x in erobot])

# panda = rtb.models.URDF.Panda()
# print(type(panda))

print_debug("full robot")
robot = robot_from_urdf(urdf_path)
links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(urdf_path, tld=".")
links_robot = [x.name for x in links]

links_head = [
    "world",
    "torso",
    "neck_dummy_link1",
    "neck_dummy_link2",
    "neck_link",
    "neck_ball_link",
    "head",
    "head_base",
    "head_tip",
]
links_l_arm = [
    "world",
    "torso",
    "l_virtual_offset",
    "l_shoulder_dummy_link",
    "l_shoulder_link",
    "l_shoulder_ball_link",
    "l_elbow_dummy_link",
    "l_elbow_link",
    "l_elbow_ball_link",
    "l_wrist_dummy_link1",
    "l_wrist_dummy_link2",
    "l_wrist_link",
    "l_wrist_ball_link",
    "l_hand_palm_link",
    "l_arm_tip",
    "l_hand_index_link",
    "l_hand_index_mimic_link",
]
links_r_arm = [
    "world",
    "torso",
    "r_virtual_offset",
    "r_shoulder_dummy_link",
    "r_shoulder_link",
    "r_shoulder_ball_link",
    "r_elbow_dummy_link",
    "r_elbow_link",
    "r_elbow_ball_link",
    "r_wrist_dummy_link1",
    "r_wrist_dummy_link2",
    "r_wrist_link",
    "r_wrist_ball_link",
    "r_hand_palm_link",
    "r_arm_tip",
    "r_hand_index_link",
    "r_hand_index_mimic_link",
]

q = np.random.rand(7)
print_debug("robot_l_arm")
robot_l_arm = robot_from_urdf(urdf_filepath, links_l_arm, gripper_name="l_arm_tip")

print_debug("robot_r_arm")
robot_r_arm = robot_from_urdf(urdf_filepath, links_r_arm, gripper_name="r_arm_tip")

print_debug("robot_head")
robot_head = robot_from_urdf(urdf_filepath, links_head, gripper_name="head_tip")
