import numpy as np
import pinocchio as pin
import PyKDL as kdl

from reachy2_modelling.kdl_parser_py import urdf
from reachy2_modelling.urdf import content as urdf_content

_, urdf_tree = urdf.treeFromString(urdf_content)
# success, urdf_tree = urdf.treeFromString(urdf_content)
# if not success:
#     raise IOError("Could not parse the URDF!")


def kdlFrame_to_np(kdlframe):
    mat = np.zeros((4, 4))
    for i in range(3):
        mat[i, 3] = kdlframe.p[i]
    for i in range(3):
        for j in range(3):
            mat[i, j] = kdlframe[i, j]
    return mat


def kdlFrame_to_SE3(X):
    return pin.SE3(kdlFrame_to_np(X))


def links_from_chain(chain):
    return [
        seg.getName()
        for seg in [chain.getSegment(x) for x in np.arange(chain.getNrOfSegments())]
    ]


def is_world_frame(chain):
    """
    if torso link exists, then it's world,
    otherwise, it's in torso frame
    """
    return "torso" in links_from_chain(chain)


def generate_solver(
    tree,
    root: str,
    tip: str,
    L: np.ndarray = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]),
):
    """Create an FK/IK solvers for each arm (left/right)."""
    chain = tree.getChain(root, tip)
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


class KDLWrapper:
    def __init__(self, tree, root: str, tip: str):
        self.chain, self.fk_solver, self.ik_solver, self.jac_solver = generate_solver(
            tree, root, tip
        )

    def link_names(self):
        return links_from_chain(self.chain)

    def print_links(self):
        for j in self.links_names():
            print(f"\t{j}")

    def joint_names(self):
        return get_chain_joints_name(self.chain)

    def print_joints(self):
        for j in self.joint_names():
            print(f"\t{j}")

    def is_world_frame(self):
        return is_world_frame(self.chain)

    def fk(self, q):
        qq = kdl.JntArray(len(q))
        for i, j in enumerate(q):
            qq[i] = j

        pose = kdl.Frame()
        _ = self.fk_solver.JntToCart(qq, pose)
        return pose

    def fk_as_SE3(self, q):
        return kdlFrame_to_SE3(self.fk(q))


r_arm_world = KDLWrapper(urdf_tree, "world", "r_arm_tip")
r_arm_torso = KDLWrapper(urdf_tree, "torso", "r_arm_tip")

l_arm_world = KDLWrapper(urdf_tree, "world", "l_arm_tip")
l_arm_torso = KDLWrapper(urdf_tree, "torso", "l_arm_tip")

head_world = KDLWrapper(urdf_tree, "world", "head_tip")
head_torso = KDLWrapper(urdf_tree, "torso", "head_tip")
