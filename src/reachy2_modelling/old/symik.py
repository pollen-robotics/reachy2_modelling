import argparse
import copy
import csv
import sys
import time
from typing import Tuple

import numpy as np
import pinocchio as pin
import PyKDL as kdl
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK
from reachy2_symbolic_ik.utils import (
    angle_diff,
    get_best_continuous_theta,
    limit_theta_to_interval,
    tend_to_prefered_theta,
)
from scipy.spatial.transform import Rotation


def inverse_kinematics(
    ik_solver, q0: np.ndarray, target_pose: np.ndarray, nb_joints: int
):
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

    def __init__(self, shoulder_offset=None):
        if shoulder_offset is None:
            shoulder_offset = [10, 0, 15]

        self.previous_theta = {}
        self.symbolic_ik_solver = {}
        self.previous_sol = {}
        self.last_call_t = {}
        self.prefered_theta = -4 * np.pi / 6  # 5 * np.pi / 4  # np.pi / 4
        self.call_timeout = 0.5
        self.orbita3D_max_angle = np.deg2rad(42.5)  # 43.5 is too much

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
                shoulder_orientation_offset=shoulder_offset,
                # This is the "wrong" values currently used by the alpha
                # shoulder_orientation_offset=[0, 0, 15],
                # shoulder_position=[-0.0479, -0.1913, 0.025],
            )
            self.last_call_t[arm] = 0

    def symbolic_inverse_kinematics(self, name, M):
        t = time.time()
        if abs(t - self.last_call_t[name]) > self.call_timeout:
            # self.logger.warning(
            #     f"{name} Timeout reached. Resetting previous_theta and previous_sol"
            # )
            self.previous_sol[name] = None
        self.last_call_t[name] = t
        d_theta_max = 0.01
        # interval_limit = [-4 * np.pi / 5, 0]
        interval_limit = [-np.pi, np.pi]

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
            ik_joints, elbow_position = theta_to_joints_func(
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
                ik_joints, elbow_position = theta_to_joints_func(
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
        # self.logger.warning(f"{name} ik={ik_joints}")
        # self.logger.warning(f"name {name} previous_sol: {self.previous_sol[name]}")

        ik_joints = self.limit_orbita3d_joints_wrist(ik_joints)
        ik_joints, multiturn = self.allow_multiturn(
            ik_joints, self.previous_sol[name], name
        )
        # self.logger.info(f"{name} ik={ik_joints}")
        self.previous_sol[name] = copy.deepcopy(ik_joints)
        # self.previous_sol[name] = ik_joints
        # self.logger.info(f"{name} ik={ik_joints}, elbow={elbow_position}")

        # TODO reactivate a smoothing technique

        return ik_joints, is_reachable, multiturn

    def allow_multiturn(self, new_joints, prev_joints, name):
        """This function will always guarantee that the joint takes the shortest path to the new position.
        The practical effect is that it will allow the joint to rotate more than 2pi if it is the shortest path.
        """
        for i in range(len(new_joints)):
            # if i == 6:
            #     self.logger.warning(f"Joint 6: [{new_joints[i]}, {prev_joints[i]}], angle_diff: {angle_diff(new_joints[i], prev_joints[i])}")
            diff = angle_diff(new_joints[i], prev_joints[i])
            new_joints[i] = prev_joints[i] + diff
        # Temp : showing a warning if a multiturn is detected. TODO do better. This info is critical and should be saved dyamically on disk.
        indexes_that_can_multiturn = [0, 2, 6]
        multiturn = False
        for index in indexes_that_can_multiturn:
            if abs(new_joints[index]) > np.pi:
                multiturn = True
                print(
                    f"Multiturn detected on joint {index} with value: {new_joints[index]} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                    f" {name} Multiturn detected on joint {index} with value: {new_joints[index]} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                )
                # TEMP forbidding multiturn
                # new_joints[index] = np.sign(new_joints[index]) * np.pi
        return new_joints, multiturn

    def limit_orbita3d_joints(self, joints):
        """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D. i.e. casting into Orbita's cone."""
        # self.logger.info(f"HEAD initial: {joints}")
        rotation = Rotation.from_euler(
            "XYZ", [joints[0], joints[1], joints[2]], degrees=False
        )
        new_joints = rotation.as_euler("ZYZ", degrees=False)
        new_joints[1] = min(
            self.orbita3D_max_angle, max(-self.orbita3D_max_angle, new_joints[1])
        )
        rotation = Rotation.from_euler("ZYZ", new_joints, degrees=False)
        new_joints = rotation.as_euler("XYZ", degrees=False)
        # self.logger.info(f"HEAD final: {new_joints}")

        return new_joints

    def limit_orbita3d_joints_wrist(self, joints):
        """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D using the wrist conventions. i.e. casting into Orbita's cone."""
        wrist_joints = joints[4:7]

        wrist_joints = self.limit_orbita3d_joints(wrist_joints)

        joints[4:7] = wrist_joints

        return joints
