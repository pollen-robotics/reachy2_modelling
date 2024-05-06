#!/usr/bin/env python

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

from ..pin import *

urdf_path = str(Path(__file__).parent / "reachy.urdf")
robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

urdf_str = open(urdf_path).read()
# model = pin.buildModelFromXML(urdf_str)

model, data = robot.model, robot.data

l_arm_joints = arm_joint_list("l")
r_arm_joints = arm_joint_list("r")
head_joints = [
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
]

# model_head = head_from_urdfstr(urdf_str)
# model_l_arm = arm_from_urdfstr(urdf_str, 'l')
# model_r_arm = arm_from_urdfstr(urdf_str, 'r')

model_l_arm, data_l_arm = model_data_from_joints(model, l_arm_joints)
model_r_arm, data_r_arm = model_data_from_joints(model, r_arm_joints)
model_head, data_head = model_data_from_joints(model, head_joints)
