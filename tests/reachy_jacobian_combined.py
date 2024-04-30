import numpy as np
import pinocchio as pin

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "models"))

import reachy_kdl as rk
import reachy_pinocchio as rp
import reachy_rtb as rr

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def separate():
    print("-" * 30)


def print_jacs(J_kdl, J_pin_lwa, J_pin_l, J_pin_w, J_rtb):
    print("J_kdl (LWA)")
    print(J_kdl)
    print("J_pin (LWA)")
    print(J_pin_lwa)
    print("J_pin (L)")
    print(J_pin_l)
    print("J_pin (W)")
    print(J_pin_w)
    print("J_rtb (?)")
    print(J_rtb)


def compute_jacs(q0, robot_rtb, jac_solver, model, data, tip):
    J_kdl = rk.jacobian(jac_solver, q0)
    joint_id = model.getFrameId(tip)
    J_pin_l = pin.computeFrameJacobian(
        model, data, q0, joint_id, reference_frame=pin.LOCAL
    )
    J_pin_lwa = pin.computeFrameJacobian(
        model, data, q0, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
    )
    J_pin_w = pin.computeFrameJacobian(
        model, data, q0, joint_id, reference_frame=pin.WORLD
    )

    # reference_frame=pin.WORLD)
    J_rtb = robot_rtb.jacobe(q0)
    print_jacs(J_kdl, J_pin_lwa, J_pin_l, J_pin_w, J_rtb)


q0 = np.random.rand(7)

#######################
separate()
print("L ARM")
jac_solver = rk.jac_solver_l_arm
model, data = rp.model_l_arm, rp.data_l_arm
# tip = 'l_wrist_yaw'
# tip = 'l_hand_palm_link'
tip = rr.robot_l_arm.ee_links_names[0]
compute_jacs(q0, rr.robot_l_arm, jac_solver, model, data, tip)

separate()
print("R ARM")
jac_solver = rk.jac_solver_r_arm
model, data = rp.model_r_arm, rp.data_r_arm
# tip = 'r_wrist_yaw'
# tip = 'r_hand_palm_link'
tip = rr.robot_r_arm.ee_links_names[0]
compute_jacs(q0, rr.robot_r_arm, jac_solver, model, data, tip)

separate()
print("HEAD")
jac_solver = rk.jac_solver_head
model, data = rp.model_head, rp.data_head
# tip = 'neck_yaw'
# tip = 'head'
tip = rr.robot_head.ee_links_names[0]
compute_jacs(q0[:3], rr.robot_head, jac_solver, model, data, tip)

separate()
print("WARNING: rtb models do not coincide with the others so far")
separate()
