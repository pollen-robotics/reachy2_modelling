#!/usr/bin/env python

import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pinocchio as pin

import reachy2_modelling as r2

original_urdf_path = str(
    Path(__file__).parent / "reachy2_rerun_test" / "reachy_v2_fix.urdf"
)
original_urdf_str = open(original_urdf_path).read()

##################################
# adapt paths
urdf_str = copy(original_urdf_str)

toreplace = "reachy_description"
urdf_str = urdf_str.replace(
    toreplace, str(Path(__file__).parent / "reachy2_rerun_test" / toreplace)
)
toreplace = "arm_description"
urdf_str = urdf_str.replace(
    toreplace, str(Path(__file__).parent / "reachy2_rerun_test" / toreplace)
)

tmp = tempfile.NamedTemporaryFile(delete=False)
urdf_path = tmp.name
with open(urdf_path, "w") as f:
    f.write(urdf_str)
##################################


def jacobian_frame(model, data, q, tip=None):
    joint_id = model.getFrameId(tip)
    return pin.computeFrameJacobian(
        model, data, q, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
    )


def jacobian_joint(q, model, data, tip):
    joint_id = model.getJointId(tip)
    J = pin.computeJointJacobian(model, data, q, joint_id)
    return J


def svals(J):
    u, s, v = np.linalg.svd(J)
    return s


def manip(J, eps=1e-6):
    det = np.linalg.det(J.T @ J)
    if det < 0 and det > -eps:
        det = 0
    return np.sqrt(det)


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


def add_framenames(model):
    model.frame_names = [x.name for x in model.frames.tolist()]


def model_from_joints(model, joints_to_keep):
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

    return model_reduced


def arm_from_urdfstr(urdfstr, arm):
    model = pin.buildModelFromXML(urdfstr)
    arm_joints = arm_joint_list(arm)
    model = model_from_joints(model, arm_joints)
    return model


def head_from_urdfstr(urdfstr):
    model = pin.buildModelFromXML(urdfstr)
    model = model_from_joints(model, head_joints)
    return model


l_arm_joints = arm_joint_list("l")
r_arm_joints = arm_joint_list("r")
head_joints = [
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
]


def offset_urdf(roll, pitch, yaw):
    new_urdf_str = copy(urdf_str)
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # larm
    tosearch = (
        '<origin rpy="-1.7453292519943295 0 -0.2617993877991494" xyz="0.0 0.2 0.0"/>'
    )
    toreplace = f'<origin rpy="{-np.pi/2 -roll} 0 {-yaw}" xyz="0.0 0.2 0.0"/>'
    new_urdf_str = new_urdf_str.replace(tosearch, toreplace)

    # rarm
    tosearch = (
        '<origin rpy="1.7453292519943295 0 0.2617993877991494" xyz="0.0 -0.2 0.0"/>'
    )
    toreplace = f'<origin rpy="{np.pi/2 + roll} 0 {yaw}" xyz="0.0 -0.2 0.0"/>'
    new_urdf_str = new_urdf_str.replace(tosearch, toreplace)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    new_urdf_file = tmp.name
    with open(new_urdf_file, "w") as f:
        f.write(new_urdf_str)
    return new_urdf_str, new_urdf_file


class Models:
    def __init__(self, urdf_str):
        self.body = pin.buildModelFromXML(urdf_str)
        self.l_arm = model_from_joints(self.body, l_arm_joints)
        self.r_arm = model_from_joints(self.body, r_arm_joints)
        self.head = model_from_joints(self.body, head_joints)


models = Models(urdf_str)


class PinArm:
    ljoints = [
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_elbow_yaw",
        "l_elbow_pitch",
        "l_wrist_roll",
        "l_wrist_pitch",
        "l_wrist_yaw",
    ]

    rjoints = [
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_elbow_yaw",
        "r_elbow_pitch",
        "r_wrist_roll",
        "r_wrist_pitch",
        "r_wrist_yaw",
    ]

    ltip = "l_arm_tip"
    rtip = "r_arm_tip"

    def __init__(self, name, model=None):
        self.arm = r2.Arm(name)
        self.prev_q = None
        self.prev_epoch_s = None

        self.model = model
        if model is None:
            if self.arm.name == "l_arm":
                self.model = models.l_arm
            else:
                self.model = models.r_arm

        self.joints = self.rjoints
        self.tip = self.rtip
        if self.arm.name == "l_arm":
            self.joints = self.ljoints
            self.tip = self.ltip

    def modeldata(self):
        return self.model, self.model.createData()

    def njoint_name(self, n):
        return self.joints[n]

    def fk(self, q, tip=None, world=False):
        if tip is None:
            tip = self.tip
        model, data = self.modeldata()
        return fk(model, data, q, tip, world)

    def jacobian(self, q, tip=None):
        if tip is None:
            tip = self.tip
        model, data = self.modeldata()
        return jacobian_frame(model, data, q, tip)

    def manip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:, :njoints]
        return manip(J)

    def linmanip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:3, :njoints]
        return manip(J)

    def ik(self, ik, M):
        return ik.symbolic_inverse_kinematics(self.arm.name, M)
