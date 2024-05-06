#!/usr/bin/env python

import sys
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pinocchio as pin

tmp = tempfile.NamedTemporaryFile()

original_urdf_path = str(
    Path(__file__).parent / "reachy2_rerun_test" / "reachy_v2_fix.urdf"
)
original_urdf_str = open(original_urdf_path).read()

urdf_str = copy(original_urdf_str)

toreplace = "reachy_description"
urdf_str = urdf_str.replace(
    toreplace, str(Path(__file__).parent / "reachy2_rerun_test" / toreplace)
)
toreplace = "arm_description"
urdf_str = urdf_str.replace(
    toreplace, str(Path(__file__).parent / "reachy2_rerun_test" / toreplace)
)
urdf_path = tmp.name
with open(tmp.name, "w") as f:
    f.write(urdf_str)

model = pin.buildModelFromXML(urdf_str)
data = model.createData()

# robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
# model, data = robot.model, robot.data


def jacobian_frame(q, modeldata, tip=None):
    [model, data] = modeldata
    if tip is None:
        tip = model.frames[-1].name
    joint_id = model.getFrameId(tip)
    J = pin.computeFrameJacobian(
        model, data, q, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
    )
    return J


def jacobian_joint(q, modeldata, tip):
    [model, data] = modeldata
    joint_id = model.getJointId(tip)
    J = pin.computeJointJacobian(model, data, q, joint_id)
    return J


def svals(J):
    u, s, v = np.linalg.svd(J)
    return s


def manip(J):
    return np.sqrt(np.linalg.det(J.T @ J))


def fk(model, data, q, tip, world_frame=False):
    # should not be needed:
    # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#a89903169c76d3c55bacaa2479bd39f76
    # pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    frame_id = model.getFrameId(tip)
    X = data.oMf[frame_id]

    # if not world, then it's torso
    if not world_frame:
        X = data.oMf[model.getFrameId("torso")].actInv(X)
    return X.copy()


def arm_joint_list(arm):
    assert arm == "l" or arm == "r"
    l_arm_joints_tokeep = [
        "_shoulder_pitch",
        "_shoulder_roll",
        "_elbow_yaw",
        "_elbow_pitch",
        "_wrist_roll",
        "_wrist_pitch",
        "_wrist_yaw",
        # '_hand_finger',
        # '_hand_finger_mimic',
    ]

    return [arm + x for x in l_arm_joints_tokeep]


l_arm_joints = arm_joint_list("l")
r_arm_joints = arm_joint_list("r")
head_joints = [
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
]


def add_framenames(model):
    model.frame_names = [x.name for x in model.frames.tolist()]


def model_data_from_joints(model, joints_to_keep):
    all_joints = model.names.tolist()
    tolock = []
    for joint in all_joints:
        if joint not in joints_to_keep and joint != "universe":
            tolock.append(joint)

    # print('to_keep:', joints_to_keep)
    # print('to_lock:', tolock)
    # Get the ID of all existing joints
    jointsToLockIDs = []
    for jn in tolock:
        # print('jn:', jn)
        if model.existJointName(jn):
            jointsToLockIDs.append(model.getJointId(jn))

    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, np.zeros(model.nq))
    add_framenames(model)

    return model_reduced, model_reduced.createData()


def arm_from_urdfstr(urdfstr, arm):
    model = pin.buildModelFromXML(urdfstr)
    arm_joints = arm_joint_list(arm)
    model, _ = model_data_from_joints(model, arm_joints)
    return model


def head_from_urdfstr(urdfstr):
    model = pin.buildModelFromXML(urdfstr)
    model, _ = model_data_from_joints(model, head_joints)
    return model


# model_head = head_from_urdfstr(urdf_str)
# model_l_arm = arm_from_urdfstr(urdf_str, 'l')
# model_r_arm = arm_from_urdfstr(urdf_str, 'r')

model_l_arm, data_l_arm = model_data_from_joints(model, l_arm_joints)
model_r_arm, data_r_arm = model_data_from_joints(model, r_arm_joints)
model_head, data_head = model_data_from_joints(model, head_joints)
