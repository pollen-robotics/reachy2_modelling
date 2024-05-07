import os
from pathlib import Path

import pandas as pd
import rerun as rr
from rerun.blueprint import (
    Blueprint,
    BlueprintPanel,
    Horizontal,
    SelectionPanel,
    Spatial3DView,
    Tabs,
    TextDocumentView,
    TimePanel,
    TimeSeriesView,
    Vertical,
)

import reachy2_modelling.pin as rp
from reachy2_modelling.old.symik import *

# ┌─────┬─────────────────────────┬───────┬───────────────────────┬──────────────────────────────────────────┐
# │link │          link           │ joint │        parent         │           ETS: parent to link            │
# ├─────┼─────────────────────────┼───────┼───────────────────────┼──────────────────────────────────────────┤
# │   0 │ world                   │       │ BASE                  │ SE3()                                    │
# │   1 │ torso                   │       │ world                 │ SE3(0, 0, 1)                             │
# │   2 │ l_virtual_offset        │       │ torso                 │ SE3(10°, -0°, 0°)                        │
# │   3 │ l_shoulder_dummy_link   │     0 │ l_virtual_offset      │ SE3(0, 0.2, 0; -10°, -0°, -15°) ⊕ Ry(q0) │
# │   4 │ l_shoulder_link         │     1 │ l_shoulder_dummy_link │ SE3() ⊕ Rx(q1)                           │
# │   5 │ l_shoulder_ball_link    │       │ l_shoulder_link       │ SE3()                                    │
# │   6 │ l_elbow_dummy_link      │     2 │ l_shoulder_ball_link  │ SE3(0, 0, -0.28; 0°, -0°, 15°) ⊕ Rz(q2)  │
# │   7 │ l_elbow_link            │     3 │ l_elbow_dummy_link    │ SE3() ⊕ Ry(q3)                           │
# │   8 │ l_elbow_ball_link       │       │ l_elbow_link          │ SE3()                                    │
# │   9 │ l_wrist_dummy_link1     │     4 │ l_elbow_ball_link     │ SE3(0, 0, -0.28; 180°, -0°, 0°) ⊕ Rx(q4) │
# │  10 │ l_wrist_dummy_link2     │     5 │ l_wrist_dummy_link1   │ SE3() ⊕ Ry(q5)                           │
# │  11 │ l_wrist_link            │     6 │ l_wrist_dummy_link2   │ SE3() ⊕ Rz(q6)                           │
# │  12 │ l_wrist_ball_link       │       │ l_wrist_link          │ SE3()                                    │
# │  13 │ @l_hand_palm_link       │       │ l_wrist_ball_link     │ SE3()                                    │
# │  14 │ l_hand_index_link       │     7 │ l_hand_palm_link      │ SE3(0, -0.03, 0.06) ⊕ Rx(q7)             │
# │  15 │ l_hand_index_mimic_link │     8 │ l_hand_palm_link      │ SE3(0, 0.03, 0.06) ⊕ Rx(q8)              │
# └─────┴─────────────────────────┴───────┴───────────────────────┴──────────────────────────────────────────┘

# >>> pin.robot.model
# Nb joints = 22 (nq=21,nv=21)
#   Joint 0 universe: parent=0
#   Joint 1 l_shoulder_pitch: parent=0
#   Joint 2 l_shoulder_roll: parent=1
#   Joint 3 l_elbow_yaw: parent=2
#   Joint 4 l_elbow_pitch: parent=3
#   Joint 5 l_wrist_roll: parent=4
#   Joint 6 l_wrist_pitch: parent=5
#   Joint 7 l_wrist_yaw: parent=6
#   Joint 8 l_hand_finger: parent=7
#   Joint 9 l_hand_finger_mimic: parent=7

#   Joint 10 neck_roll: parent=0
#   Joint 11 neck_pitch: parent=10
#   Joint 12 neck_yaw: parent=11
#   Joint 13 r_shoulder_pitch: parent=0
#   Joint 14 r_shoulder_roll: parent=13
#   Joint 15 r_elbow_yaw: parent=14
#   Joint 16 r_elbow_pitch: parent=15
#   Joint 17 r_wrist_roll: parent=16
#   Joint 18 r_wrist_pitch: parent=17
#   Joint 19 r_wrist_yaw: parent=18
#   Joint 20 r_hand_finger: parent=19
#   Joint 21 r_hand_finger_mimic: parent=19


def series_to_target_mat(series):
    target_pose_cols = ["pos_x", "pos_y", "pos_z", "or_x", "or_y", "or_z", "or_w"]
    lcols = [x + "_left" for x in target_pose_cols]
    rcols = [x + "_right" for x in target_pose_cols]

    def coords_array_to_mat(coords):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(coords[3:]).as_matrix()
        M[:3, 3] = coords[:3]
        return M

    def series_to_mat(myseries, cols):
        arm_series = myseries[cols]
        hasnan = arm_series.isnull().values.any()
        if hasnan:
            return None
        return coords_array_to_mat(arm_series.values)

    Ml = series_to_mat(series, lcols)
    Mr = series_to_mat(series, rcols)

    return Ml, Mr


def df_target_poses(larmf, rarmf):
    error = False
    for armf in [larmf, rarmf]:
        armf_found = os.path.isfile(armf)
        if not armf_found:
            print(f"Error: {armf} not found")
            error = True
    if error:
        exit(1)

    dfl = pd.read_csv(larmf)
    dfl = dfl.set_index("epoch_s", verify_integrity=True)
    dfr = pd.read_csv(rarmf)
    dfr = dfr.set_index("epoch_s", verify_integrity=True)

    return dfl.join(dfr, lsuffix="_left", rsuffix="_right", how="outer")


class ArmHandler:
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

    def __init__(self, name, model):
        self.name = name
        assert self.name == "l_arm" or self.name == "r_arm"
        self.prev_q = None
        self.prev_epoch_s = None
        self.model = model

        self.joints = self.rjoints
        self.tip = self.rtip
        if self.name == "l_arm":
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
        return rp.fk(model, data, q, tip, world)

    def jacobian(self, q, tip=None):
        if tip is None:
            tip = self.tip
        model, data = self.modeldata()
        return rp.jacobian_frame(model, data, q, tip)

    def manip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:, :njoints]
        return rp.manip(J)

    def linmanip(self, q, tip, njoints=None):
        if njoints is None:
            njoints = 7
        J = self.jacobian(q, tip)[:3, :njoints]
        return rp.manip(J)

    def log(self, epoch_s, ik, M, urdf_logger):
        if M is not None:
            q, reachable, multiturn = ik.symbolic_inverse_kinematics(self.name, M)

            for joint_idx, angle in enumerate(q):
                urdf_logger.log_joint_angle(self.njoint_name(joint_idx), angle)

            if self.prev_q is not None:
                dt = epoch_s - self.prev_epoch_s
                qd = (q - self.prev_q) / dt
            else:
                dt = 0
                qd = np.zeros_like(q)

            self.prev_q = q
            self.prev_epoch_s = epoch_s

            # manipulability
            for dof in [2, 4, 7]:
                tip = self.joints[dof - 1]
                rr.log(
                    arm_entity(self.name, f"manip/{dof}dof"),
                    rr.Scalar(self.manip(q, tip, njoints=dof)),
                )
                rr.log(
                    arm_entity(self.name, f"linmanip/{dof}dof"),
                    rr.Scalar(self.manip(q, tip, njoints=dof)),
                )

            # fk to compute error
            Mcur = self.fk(q)
            trans, R = Mcur.translation, Mcur.rotation
            for i, coord in enumerate(["x", "y", "z"]):
                entity = teleop_arm_entity(self.name, f"state_{coord}")
                rr.log(entity, rr.Scalar(trans[i]))

            # sym ik stats
            rr.log(arm_entity(self.name, "ik/reachable"), rr.Scalar(reachable))
            rr.log(arm_entity(self.name, "ik/multiturn"), rr.Scalar(multiturn))
            rr.log(arm_entity(self.name, "ik/dt"), rr.Scalar(dt))

            # joint states
            for j, ang in enumerate(q):
                rr.log(arm_q_entity(self.name, j), rr.Scalar(ang))
            for j, vel in enumerate(qd):
                rr.log(arm_qd_entity(self.name, j), rr.Scalar(vel))


class Scene:
    # Example based on
    # https://github.com/rerun-io/python-example-droid-dataset/
    dir_path: Path

    def __init__(self, dir_path: Path, model_l_arm, model_r_arm):
        self.dir_path = dir_path

        larmf = os.path.join(dir_path, "l_arm_target_pose.csv")
        rarmf = os.path.join(dir_path, "r_arm_target_pose.csv")
        print("csv files:", larmf, rarmf)
        self.df = df_target_poses(larmf, rarmf)

        self.ik = MySymIK()
        self.larm = ArmHandler("l_arm", model_l_arm)
        self.rarm = ArmHandler("r_arm", model_r_arm)

    def log_teleop(self, Ml, Mr, torso_entity):
        def pub(arm, M):
            entity_base = f"{torso_entity}/{arm}_"
            if M is not None:
                trans, R = M[:3, 3], M[:3, :3]
                entity = entity_base + "target_pose"
                rr.log(entity, rr.Transform3D(translation=trans, mat3x3=R))
                entity = entity_base + "target_cartesian_position"
                rr.log(entity, rr.Points3D(trans, radii=0.02))
                for i, coord in enumerate(["x", "y", "z"]):
                    entity = teleop_arm_entity(arm, f"target_{coord}")
                    rr.log(entity, rr.Scalar(trans[i]))

        pub("l_arm", M=Ml)
        pub("r_arm", M=Mr)

    def log(self, urdf_logger) -> None:

        first_step = True
        for index_series in self.df.iterrows():
            (epoch_s, series) = index_series
            rr.set_time_seconds("real_time", epoch_s)

            if first_step:
                # We want to log the robot model here so that it appears in the right timeline
                print("first step urdf_logger.log()")
                urdf_logger.log()
                first_step = False
                # exit(0)

            Ml, Mr = series_to_target_mat(series)

            self.log_teleop(Ml, Mr, urdf_logger.torso_entity)
            self.larm.log(epoch_s, self.ik, Ml, urdf_logger)
            self.rarm.log(epoch_s, self.ik, Mr, urdf_logger)


def teleop_arm_entity(name, i):
    return f"teleop_{name}/{i}"


def arm_entity(name, i):
    return f"/{name}/{i}"


def arm_q_entity(name, i):
    return arm_entity(name, "q") + f"{i}"


def arm_qd_entity(name, i):
    return arm_entity(name, "qd") + f"{i}"


def teleop_blueprint():
    return Tabs(
        Horizontal(
            TimeSeriesView(origin=teleop_arm_entity("l_arm", "")),
            TimeSeriesView(origin=teleop_arm_entity("r_arm", "")),
            name="teleop_position",
        ),
        Horizontal(
            Vertical(
                TimeSeriesView(origin=arm_entity("l_arm", "ik/reachable")),
                TimeSeriesView(origin=arm_entity("l_arm", "ik/multiturn")),
                TimeSeriesView(origin=arm_entity("l_arm", "ik/dt")),
            ),
            Vertical(
                TimeSeriesView(origin=arm_entity("r_arm", "ik/reachable")),
                TimeSeriesView(origin=arm_entity("r_arm", "ik/multiturn")),
                TimeSeriesView(origin=arm_entity("r_arm", "ik/dt")),
            ),
            name="IK Stats",
        ),
        active_tab=1,
    )


def arm_joints_tab(name):

    return Horizontal(
        Vertical(
            *(TimeSeriesView(origin=arm_q_entity(name, i)) for i in range(7)),
            name="Q",
        ),
        Vertical(
            *(TimeSeriesView(origin=arm_qd_entity(name, i)) for i in range(7)),
            name="Qd",
        ),
        name=name,
    )


def debug_tab():

    return Vertical(
        Horizontal(
            TimeSeriesView(origin=arm_entity("l_arm", f"manip/")),
            TimeSeriesView(origin=arm_entity("r_arm", f"manip/")),
            name="full manipulability",
        ),
        Horizontal(
            TimeSeriesView(origin=arm_entity("l_arm", f"linmanip/")),
            TimeSeriesView(origin=arm_entity("r_arm", f"linmanip/")),
            name="linear manipulability",
        ),
        Horizontal(
            Vertical(
                TimeSeriesView(
                    origin=arm_entity("l_arm", ""),
                    contents=[f"$origin/qd{i}" for i in range(2)],
                ),
                TimeSeriesView(
                    origin=arm_entity("l_arm", ""),
                    contents=[f"$origin/qd{i+2}" for i in range(2)],
                ),
            ),
            Vertical(
                TimeSeriesView(
                    origin=arm_entity("r_arm", ""),
                    contents=[f"$origin/qd{i}" for i in range(2)],
                ),
                TimeSeriesView(
                    origin=arm_entity("r_arm", ""),
                    contents=[f"$origin/qd{i+2}" for i in range(2)],
                ),
            ),
            name="Qd2",
        ),
        name="debug",
        row_shares=[1, 1, 2],
    )


def blueprint():
    return Blueprint(
        Horizontal(
            Vertical(
                Spatial3DView(name="spatial view", origin="/", contents=["/**"]),
                teleop_blueprint(),
                row_shares=[1, 1],
            ),
            Vertical(
                Tabs(
                    arm_joints_tab("l_arm"),
                    arm_joints_tab("r_arm"),
                    debug_tab(),
                    active_tab=2,
                ),
            ),
            column_shares=[1.3, 2],
        ),
        BlueprintPanel(expanded=False),
        SelectionPanel(expanded=False),
        TimePanel(expanded=False),
    )

    # return Blueprint(
    #     Horizontal(
    #         Vertical(
    #             Spatial3DView(name="spatial view", origin="/", contents=["/**"]),
    #             blueprint_row_images(
    #                 [
    #                     f"/cameras/{cam}"
    #                         for cam in [
    #                                 "exterior_image_1_left",
    #                                 "exterior_image_2_left",
    #                                 "wrist_image_left",
    #                         ]
    #                 ]
    #             ),
    #             row_shares=[3, 1],
    #         ),
    #         Vertical(
    #             Tabs( # Tabs for all the different time serieses.
    #                 Vertical(
    #                     *(
    #                         TimeSeriesView(origin=f"/action_dict/joint_velocity/{i}")
    #                             for i in range(7)
    #                     ),
    #                     name="joint velocity",
    #                 ),
    #                 Vertical(
    #                     *(
    #                         TimeSeriesView(origin=f"/action_dict/cartesian_velocity/{i}")
    #                             for i in range(6)
    #                     ),
    #                     name="cartesian position",
    #                 ),
    #                 Vertical(
    #                     TimeSeriesView(origin="/action_dict/gripper_position"),
    #                     TimeSeriesView(origin="/action_dict/gripper_velocity"),
    #                     name="gripper",
    #                 ),
    #                 TimeSeriesView(origin="/discount"),
    #                 TimeSeriesView(origin="/reward"),
    #                 active_tab=0,
    #             ),
    #             TextDocumentView(origin='instructions'),
    #             row_shares=[7, 1]
    #         ),
    #         column_shares=[3, 1],
    #     ),
    #     SelectionPanel(expanded=False),
    #     TimePanel(expanded=False),
    # )
