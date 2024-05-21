from pathlib import Path

import numpy as np
import PyKDL as kdl

from ..kdl_parser_py import urdf

urdf_path = str(Path(__file__).parent / "reachy.urdf")

urdf_content = ""
with open(urdf_path, "r") as f:
    urdf_content = f.read()


def print_debug(*args):
    pass
    # print(*args)


def links_from_chain(chain):
    return [
        seg.getName()
        for seg in [chain.getSegment(x) for x in np.arange(chain.getNrOfSegments())]
    ]


def world_frame(chain):
    # if torso link exists, then it's world,
    # otherwise, it's in torso frame
    return "torso" in links_from_chain(chain)


def generate_solver(
    urdf_str: str,
    root: str,
    tip: str,
    L: np.ndarray = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]),
):
    """Create an FK/IK solvers for each arm (left/right)."""
    success, urdf_tree = urdf.treeFromString(urdf_str)
    if not success:
        raise IOError("Could not parse the URDF!")

    chain = urdf_tree.getChain(root, tip)
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)

    ik_solver = kdl.ChainIkSolverPos_LMA(
        chain,
        eps=1e-5,
        # TODO: probably different PyKDL version
        # code tested in ubuntu 20.04 but comes from 22.04
        # maxiter=500,
        # eps_joints=1e-15,
        # L=L,
    )

    jac_solver = kdl.ChainJntToJacSolver(chain)

    return chain, fk_solver, ik_solver, jac_solver


def get_chain_joints_name(chain):
    joints = []

    for i in range(chain.getNrOfSegments()):
        joint = chain.getSegment(i).getJoint()
        if joint.getType() == joint.RotAxis:
            joints.append(joint.getName())

    return joints


def jacobian(jac_solver, myotherq):
    # compute jac
    jkdl = kdl.Jacobian(len(myotherq))
    qkdl = kdl.JntArray(len(myotherq))
    for i, j in enumerate(myotherq):
        qkdl[i] = j
    jac_solver.JntToJac(qkdl, jkdl)

    # kdl.Jacobian -> np.array
    myjac = np.zeros((jkdl.rows(), jkdl.columns()))
    for i in range(jkdl.rows()):
        for j in range(jkdl.columns()):
            myjac[i, j] = jkdl[i, j]
    return myjac


chain_r_arm, fk_solver_r_arm, ik_solver_r_arm, jac_solver_r_arm = generate_solver(
    urdf_content, "world", "r_arm_tip"
)
(
    chain_r_arm_torso,
    fk_solver_r_arm_torso,
    ik_solver_r_arm_torso,
    jac_solver_r_arm_torso,
) = generate_solver(urdf_content, "torso", "r_arm_tip")
for j in get_chain_joints_name(chain_r_arm):
    print(f"\t{j}")

chain_l_arm, fk_solver_l_arm, ik_solver_l_arm, jac_solver_l_arm = generate_solver(
    urdf_content, "world", "l_arm_tip"
)
(
    chain_l_arm_torso,
    fk_solver_l_arm_torso,
    ik_solver_l_arm_torso,
    jac_solver_l_arm_torso,
) = generate_solver(urdf_content, "torso", "l_arm_tip")

for j in get_chain_joints_name(chain_l_arm):
    print(f"\t{j}")
q0 = [0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]
print_debug("left q0:", q0)
q0 = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]
print_debug("right q0:", q0)

# Kinematics for the head
chain_head, fk_solver_head, ik_solver_head, jac_solver_head = generate_solver(
    urdf_content,
    "world",
    "head_tip",
    L=np.array([1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0]),
)  # L weight matrix to considere only the orientation

chain_head_torso, fk_solver_head_torso, ik_solver_head_torso, jac_solver_head_torso = (
    generate_solver(
        urdf_content,
        "torso",
        "head_tip",
        L=np.array([1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0]),
    )
)  # L weight matrix to considere only the orientation

links_l_arm = links_from_chain(chain_l_arm)
links_r_arm = links_from_chain(chain_r_arm)

links_head = links_from_chain(chain_head)
