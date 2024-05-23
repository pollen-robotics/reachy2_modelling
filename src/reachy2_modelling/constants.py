import numpy as np

head_links = [
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

head_joints = [
    "neck_roll",
    "neck_pitch",
    "neck_yaw",
]

l_arm_links = [
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

l_arm_joints = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow_yaw",
    "l_elbow_pitch",
    "l_wrist_roll",
    "l_wrist_pitch",
    "l_wrist_yaw",
]

r_arm_joints = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_elbow_yaw",
    "r_elbow_pitch",
    "r_wrist_roll",
    "r_wrist_pitch",
    "r_wrist_yaw",
]


r_arm_links = [
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


l_arm_tip = "l_arm_tip"
r_arm_tip = "r_arm_tip"
head_tip = "head_tip"


left_q0 = [0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]
right_q0 = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]
