# general_robotics_toolbox.urdf.robot_from_xml_string
import sys

import pinocchio as pin
import roboticstoolbox as rtb

import reachy2_modelling as r2
from reachy2_modelling.urdf import path_old as urdf_path


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


class RTBWrapper:
    def __init__(self, robot):
        self.robot = robot

    def fk_as_SE3(self, q, tip, world_frame=False):
        if world_frame:
            X = pin.SE3(self.robot.fkine(q, end=tip).A)
        else:
            Xworld_torso = self.robot.fkine(q, end="torso")
            Xworld_ee = self.robot.fkine(q)
            X = pin.SE3((Xworld_torso.inv() * Xworld_ee).A)
            X = pin.SE3((self.robot.fkine(q, start="torso", end=tip)).A)
        return X


fullrobot = robot_from_urdf(urdf_path)

l_arm = RTBWrapper(
    robot_from_urdf(
        urdf_path, r2.constants.l_arm_links, gripper_name=r2.constants.l_arm_tip
    )
)
r_arm = RTBWrapper(
    robot_from_urdf(
        urdf_path, r2.constants.r_arm_links, gripper_name=r2.constants.r_arm_tip
    )
)
head = RTBWrapper(
    robot_from_urdf(
        urdf_path, r2.constants.head_links, gripper_name=r2.constants.head_tip
    )
)
