import tempfile
from copy import copy

import numpy as np
import pinocchio as pin

import reachy2_modelling as r2


def jacobian_frame(model, data, q, tip=None):
    joint_id = model.getFrameId(tip)
    return pin.computeFrameJacobian(
        model, data, q, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
    )


def jacobian_joint(q, model, data, tip):
    joint_id = model.getJointId(tip)
    J = pin.computeJointJacobian(model, data, q, joint_id)
    return J


def fk(model, data, q, tip, torso_frame=False):
    # should not be needed:
    # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#a89903169c76d3c55bacaa2479bd39f76
    # pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    # pin.framesForwardKinematics(model, data, q)

    frame_id = model.getFrameId(tip)
    X = data.oMf[frame_id]

    # if not world, then it's torso
    if torso_frame:
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


def shoulder_offset_urdf(roll, pitch, yaw):
    new_urdf_str = copy(r2.urdf.content)
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


class PinModels:
    @staticmethod
    def from_shoulder_offset(roll, pitch, yaw):
        urdf_str, urdf_path = shoulder_offset_urdf(roll, pitch, yaw)
        return PinModels(urdf_str), urdf_str, urdf_path

    def __init__(self, urdf_str):
        self.body = pin.buildModelFromXML(urdf_str)
        self.l_arm = model_from_joints(self.body, l_arm_joints)
        self.r_arm = model_from_joints(self.body, r_arm_joints)
        self.head = model_from_joints(self.body, head_joints)


models = PinModels(r2.urdf.content)


class PinWrapper:
    def __init__(self, model, tip, joints, urdf_path=None, urdf_str=None):
        self.model = model
        self.tip = tip
        self.joints = joints
        self.urdf_path = urdf_path
        self.urdf_str = urdf_str

    def modeldata(self):
        return self.model, self.model.createData()

    def model_collision(self):
        assert self.urdf_str is not None
        return pin.buildGeomFromUrdfString(
            self.model, self.urdf_str, pin.COLLISION
        )  # , package_dirs=package_dirs)

    def model_visual(self):
        assert self.urdf_str is not None
        return pin.buildGeomFromUrdfString(
            self.model, self.urdf_str, pin.VISUAL
        )  # , package_dirs=package_dirs)

    def njoint_name(self, n):
        return self.joints[n]

    def fk(self, q, tip=None, torso_frame=False):
        if tip is None:
            tip = self.tip
        model, data = self.modeldata()
        return fk(model, data, q, tip, torso_frame)

    def jacobian(self, q, tip=None):
        if tip is None:
            tip = self.tip
        model, data = self.modeldata()
        return jacobian_frame(model, data, q, tip)

    def manip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:, :njoints]
        return r2.algo.manip(J)

    def linmanip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:3, :njoints]
        return r2.algo.manip(J)


class PinWrapperArm(PinWrapper):
    @staticmethod
    def from_shoulder_offset(name, roll, pitch, yaw):
        models, urdf_str, urdf_path = r2.pin.PinModels.from_shoulder_offset(
            roll, pitch, yaw
        )
        model = models.r_arm
        if name == "l_arm":
            model = models.l_arm
        return PinWrapperArm(
            name, custom_model=model, urdf_str=urdf_str, urdf_path=urdf_path
        )

    def __init__(self, name, custom_model=None, urdf_str=None, urdf_path=None):
        self.arm = r2.Arm(name)

        model = models.r_arm
        joints = r2.constants.r_arm_joints
        tip = r2.constants.r_arm_tip
        if name == "l_arm":
            model = models.l_arm
            joints = r2.constants.l_arm_joints
            tip = r2.constants.l_arm_tip

        if custom_model is not None:
            model = custom_model
        super().__init__(
            model=model, tip=tip, joints=joints, urdf_str=urdf_str, urdf_path=urdf_path
        )


class PinWrapperHead(PinWrapper):
    def __init__(self, custom_model=None, urdf_str=None, urdf_path=None):
        model = models.head
        joints = r2.constants.head_joints
        tip = r2.constants.head_tip
        if custom_model is not None:
            model = custom_model
        super().__init__(
            model=model, tip=tip, joints=joints, urdf_str=urdf_str, urdf_path=urdf_path
        )


def wrappers_from_shoulder_offset(roll, pitch, yaw):
    urdf_str, urdf_path = shoulder_offset_urdf(roll, pitch, yaw)
    modmodels = PinModels(urdf_str)
    return (
        PinWrapperArm("l_arm", modmodels.l_arm, urdf_str=urdf_str, urdf_path=urdf_path),
        PinWrapperArm("r_arm", modmodels.l_arm, urdf_str=urdf_str, urdf_path=urdf_path),
        PinWrapperHead(modmodels.head, urdf_str=urdf_str, urdf_path=urdf_path),
        urdf_str,
        urdf_path,
    )


urdf_str = r2.urdf.content
urdf_path = r2.urdf.path
l_arm = PinWrapperArm("l_arm", urdf_str=urdf_str, urdf_path=urdf_path)
r_arm = PinWrapperArm("r_arm", urdf_str=urdf_str, urdf_path=urdf_path)
head = PinWrapperHead(urdf_str=urdf_str, urdf_path=urdf_path)
