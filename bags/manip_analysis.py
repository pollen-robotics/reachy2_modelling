#!/usr/bin/env python3

import argparse
import csv
import sys
from typing import Tuple

import numpy as np
import pinocchio as pin
import PyKDL as kdl
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (angle_diff, get_best_continuous_theta,
                                       limit_theta_to_interval,
                                       tend_to_prefered_theta)
from scipy.spatial.transform import Rotation

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

urdf_path = "../reachy.urdf"
robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
model, data = robot.model, robot.data
# lock the other joints
tolock = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow_yaw",
    "l_elbow_pitch",
    "l_wrist_roll",
    "l_wrist_pitch",
    "l_wrist_yaw",
    "l_hand_finger",
    "l_hand_finger_mimic",
    # 'r_wrist_roll',
    # 'r_wrist_pitch',
    # 'r_wrist_yaw',
    "r_hand_finger",
    "r_hand_finger_mimic",
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
]

# tolock = [
#     'r_shoulder_pitch',
#     'r_shoulder_roll',
#     'r_elbow_yaw',
#     'r_elbow_pitch',
#     'r_wrist_roll',
#     'r_wrist_pitch',
#     'r_wrist_yaw',
#     'r_hand_finger',
#     'r_hand_finger_mimic',
#     # 'l_wrist_roll',
#     # 'l_wrist_pitch',
#     # 'l_wrist_yaw',
#     'l_hand_finger',
#     'l_hand_finger_mimic',
#     'neck_roll',
#     'neck_pitch',
#     'neck_yaw',
# ]

# Get the ID of all existing joints
jointsToLockIDs = []
for jn in tolock:
    if model.existJointName(jn):
        jointsToLockIDs.append(model.getJointId(jn))
robot.model = pin.buildReducedModel(model, jointsToLockIDs, np.zeros(21))
robot.data = robot.model.createData()


def jacobian_frame(q, tip=None, robot=robot):
    if tip is None:
        tip = robot.model.frames[-1].name
    joint_id = robot.model.getFrameId(tip)
    J = pin.computeFrameJacobian(
        robot.model, robot.data, q, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
    )
    return J


def jacobian_joint(q, tip=None, robot=robot):
    joint_id = robot.model.getJointId(tip)
    J = pin.computeJointJacobian(robot.model, robot.data, q, joint_id)
    return J


def svals(J):
    u, s, v = np.linalg.svd(J)
    return s


def manip(J):
    return np.sqrt(np.linalg.det(J.T @ J))


def fk(q, tip=None, robot=robot):
    # joint_id =  robot.model.getFrameId(robot.model.frames[-1].name)
    if tip is None:
        tip = robot.model.frames[-1].name
    joint_id = robot.model.getFrameId(tip)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    # pin.computeJointJacobians(robot.model, robot.data, q)
    return robot.data.oMf[robot.model.getFrameId(tip)].copy()


def inverse_kinematics(
    ik_solver, q0: np.ndarray, target_pose: np.ndarray, nb_joints: int
) -> Tuple[float, np.ndarray]:
    """Compute the inverse kinematics of the given arm.
    The function assumes the number of joints is correct!
    """
    x, y, z = target_pose[:3, 3]
    R = target_pose[:3, :3].flatten().tolist()

    _q0 = kdl.JntArray(nb_joints)
    for i, q in enumerate(q0):
        _q0[i] = q

    pose = kdl.Frame()
    pose.p = kdl.Vector(x, y, z)
    pose.M = kdl.Rotation(*R)

    sol = kdl.JntArray(nb_joints)
    res = ik_solver.CartToJnt(_q0, pose, sol)
    sol = list(sol)

    return res, sol


def get_euler_from_homogeneous_matrix(homogeneous_matrix, degrees: bool = False):
    position = homogeneous_matrix[:3, 3]
    rotation_matrix = homogeneous_matrix[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler(
        "xyz", degrees=degrees
    )
    return position, euler_angles


class MySymIK:

    def __init__(self):
        self.previous_theta = {}
        self.symbolic_ik_solver = {}
        self.previous_sol = {}
        self.prefered_theta = -4 * np.pi / 6  # 5 * np.pi / 4  # np.pi / 4

        for arm in ["l_arm", "r_arm"]:
            self.previous_theta[arm] = None
            self.previous_sol[arm] = None
            self.symbolic_ik_solver[arm] = SymbolicIK(
                arm=arm,
                upper_arm_size=0.28,
                forearm_size=0.28,
                gripper_size=0.10,
                wrist_limit=45,
                # This is the "correct" stuff for alpha
                shoulder_orientation_offset=[10, 0, 15],
                # This is the "wrong" values currently used by the alpha
                # shoulder_orientation_offset=[0, 0, 15],
                # shoulder_position=[-0.0479, -0.1913, 0.025],
            )

    def symbolic_inverse_kinematics(self, name, M):
        d_theta_max = 0.01
        interval_limit = [-4 * np.pi / 5, 0]

        if name.startswith("r"):
            prefered_theta = self.prefered_theta
        else:
            prefered_theta = -np.pi - self.prefered_theta

        if name.startswith("l"):
            interval_limit = [-np.pi - interval_limit[1], -np.pi - interval_limit[0]]

        if self.previous_theta[name] is None:
            self.previous_theta[name] = prefered_theta

        if self.previous_sol[name] is None:
            self.previous_sol[name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # self.logger.warning(
        #     f"{name} prefered_theta: {prefered_theta}, previous_theta: {self.previous_theta[name]}"
        # )

        goal_position, goal_orientation = get_euler_from_homogeneous_matrix(M)

        # self.logger.warning(f"{name} goal_position: {goal_position}")

        goal_pose = np.array([goal_position, goal_orientation])

        is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[
            name
        ].is_reachable(goal_pose)
        if is_reachable:
            is_reachable, theta, state = get_best_continuous_theta(
                self.previous_theta[name],
                interval,
                theta_to_joints_func,
                d_theta_max,
                prefered_theta,
                self.symbolic_ik_solver[name].arm,
            )
            # self.logger.warning(
            #    f"name: {name}, theta: {theta}")
            theta = limit_theta_to_interval(
                theta, self.previous_theta[name], interval_limit
            )
            # self.logger.warning(
            #    f"name: {name}, theta: {theta}, previous_theta: {self.previous_theta[name]}, state: {state}"
            # )
            self.previous_theta[name] = theta
            self.ik_joints, elbow_position = theta_to_joints_func(
                theta, previous_joints=self.previous_sol[name]
            )
            # self.logger.warning(
            #    f"{name} Is reachable. Is truly reachable: {is_reachable}. State: {state}"
            # )

        else:
            # self.logger.warning(f"{name} Pose not reachable but doing our best")
            is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[
                name
            ].is_reachable_no_limits(goal_pose)
            if is_reachable:
                is_reachable, theta = tend_to_prefered_theta(
                    self.previous_theta[name],
                    interval,
                    theta_to_joints_func,
                    d_theta_max,
                    goal_theta=prefered_theta,
                )
                theta = limit_theta_to_interval(
                    theta, self.previous_theta[name], interval_limit
                )
                # self.logger.warning(
                #    f"name: {name}, theta: {theta}, previous_theta: {self.previous_theta[name]}"
                # )
                self.previous_theta[name] = theta
                self.ik_joints, elbow_position = theta_to_joints_func(
                    theta, previous_joints=self.previous_sol[name]
                )
            else:
                print(
                    f"{name} Pose not reachable, this has to be fixed by projecting far poses to reachable sphere"
                )
                raise RuntimeError(
                    "Pose not reachable in symbolic IK. We crash on purpose while we are on the debug sessions. This piece of code should disapiear after that."
                )
        # self.logger.warning(f"{name} new_theta: {theta}")
        # if name.startswith("l"):
        #     self.logger.warning(
        #         f"Symetrised previous_theta diff: {(self.previous_theta['r_arm'] - (np.pi - self.previous_theta['l_arm']))%(2*np.pi)}"
        #     )

        # self.logger.warning(f"{name} jump in joint space")

        self.ik_joints, multiturn = self.allow_multiturn(
            self.ik_joints, self.previous_sol[name]
        )

        self.previous_sol[name] = self.ik_joints
        # self.logger.info(f"{name} ik={self.ik_joints}, elbow={elbow_position}")

        # TODO reactivate a smoothing technique

        return self.ik_joints, is_reachable, multiturn

    def allow_multiturn(self, new_joints, prev_joints):
        """This function will always guarantee that the joint takes the shortest path to the new position.
        The practical effect is that it will allow the joint to rotate more than 2pi if it is the shortest path.
        """
        for i in range(len(new_joints)):
            diff = angle_diff(new_joints[i], prev_joints[i])
            new_joints[i] = prev_joints[i] + diff

        # Temp : showing a warning if a multiturn is detected. TODO do better. This info is critical and should be saved dyamically on disk.
        indexes_that_can_multiturn = [0, 2, 6]
        multiturn = False
        for index in indexes_that_can_multiturn:
            if abs(new_joints[index]) > np.pi:
                multiturn = True
                # print(
                #     f"Multiturn detected on joint {index} with value: {new_joints[index]} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                # )
                # TEMP forbidding multiturn
                # new_joints[index] = np.sign(new_joints[index]) * np.pi
        return new_joints, multiturn


parser = argparse.ArgumentParser()
parser.add_argument("csvfile", type=str)
args = parser.parse_args()

ik = MySymIK()
print("csvfile:", args.csvfile)

csvfilebase = args.csvfile[:-4]
outcsvfile = "{}_symik.csv".format(csvfilebase)

print("computing manip to csv...")
with open(args.csvfile, newline="") as csvfile:
    i = 0
    reader = csv.DictReader(csvfile)
    for row in reader:
        i += 1
        # (x, y, z, w)
        R = Rotation.from_quat([row["or_x"], row["or_y"], row["or_z"], row["or_w"]])
        M = np.eye(4)
        M[:3, :3] = R.as_matrix()
        M[:3, 3] = np.array(
            [
                row["pos_x"],
                row["pos_y"],
                row["pos_z"],
            ]
        )
        # print(M)
        q, reachable, multiturn = ik.symbolic_inverse_kinematics("l_arm", M)

        # tip = 'r_elbow_ball_link' # as frame
        # tip = 'r_elbow_dummy_link'
        # tip = 'r_shoulder_ball_link'
        tip = "r_elbow_pitch"  # joint
        # tip = None
        fkk = fk(q, tip=tip)
        # J = jacobian_frame(q, tip=tip)[:3, :]
        J = jacobian_joint(q, tip=tip)[:3, :]
        manipp = manip(J)  # TODO: NOT WORKING
        rank = np.linalg.matrix_rank(J)
        svalues = svals(J)
        nsvalues = 2

        # if manipp > 0.1:
        #     print_all()
        #     sys.exit(0)

        def print_all():
            print("rank", rank)
            print("multiturn:", multiturn)
            print("q:", q)
            print("svalues:", svalues)
            print("manip: {:.10f}".format(manipp))
            print(fkk)
            print(J)

        if np.min(svalues[:nsvalues]) < 0.22:
            print("-------------")
            print("svalues LOWW i:", i)
            print_all()

        if rank != nsvalues:
            print("------------------")
            print("rank != J.shape[0] i:", i)
            print_all()

        if multiturn:
            print("-------------")
            print("multiturn detected!, i:", i)
            print_all()
