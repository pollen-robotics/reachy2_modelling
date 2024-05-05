import numpy as np
import pinocchio as pin
import PyKDL as kdl
from example_robot_data import load

from pathlib import Path

try:
    import reachy2_modelling
    from reachy2_modelling.kdl_parser_py import urdf
except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)


np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def separate():
    print("-" * 30)


def separate_shmol():
    print("-" * 10)


def kdl_jacobian(jac_solver, myotherq):
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


# the file in this repo was extracted from example_robot_data
panda_path = str(Path(reachy2_modelling.__file__).parent / "panda.urdf")
# robot = pin.RobotWrapper.BuildFromURDF(panda_path)
robot = load("panda")
model, data = robot.model, robot.data
# tip = 'panda_joint7'
tip = "panda_link7"
joint_id = robot.model.getFrameId(tip)

with open(panda_path, "r") as f:
    panda_content = f.read()
chain, fk_solver, ik_solver, jac_solver = generate_solver(
    # panda_content, 'universe', tip)
    panda_content,
    "panda_link0",
    tip,
)

q0 = np.random.rand(9)

separate()
print("JACOBIANS")
# jac_pin = pin.computeJointJacobian(model, data, q0, joint_id)
jac_pin = pin.computeFrameJacobian(model, data, q0, joint_id)[:, :7]
jac_pin_local = pin.computeFrameJacobian(
    model, data, q0, joint_id, reference_frame=pin.LOCAL
)[:, :7]
jac_pin_lwa = pin.computeFrameJacobian(
    model, data, q0, joint_id, reference_frame=pin.LOCAL_WORLD_ALIGNED
)[:, :7]
jac_pin_wor = pin.computeFrameJacobian(
    model, data, q0, joint_id, reference_frame=pin.WORLD
)[:, :7]
jac_kdl = kdl_jacobian(jac_solver, q0[:7])

print("J pin default (local)", jac_pin)
separate_shmol()
print("J loc", jac_pin_local)
separate_shmol()
print("J lwa", jac_pin_lwa)
separate_shmol()
print("J wor", jac_pin_wor)
separate_shmol()
print("J kdl", jac_kdl)

val = [1, 2, 3]
R_pin = pin.rpy.rpyToMatrix(*val)
R_kdl = kdl.Rotation.RPY(*val)
print("R_pin", R_pin)
print("R_kdl", R_kdl)

separate()
print("FRAMES")
X0_pin = pin.SE3.Random()
X1_pin = pin.SE3.Random()
print("X0_pin", X0_pin)
print("X1_pin", X1_pin)

X0_kdl = kdl.Frame(
    kdl.Rotation(*X0_pin.rotation.ravel().tolist()), kdl.Vector(*X0_pin.translation)
)
X1_kdl = kdl.Frame(
    kdl.Rotation(*X1_pin.rotation.ravel().tolist()), kdl.Vector(*X1_pin.translation)
)
print("X0_kdl.M", X0_kdl.M)
print("X0_kdl.p", X0_kdl.p)
print("X1_kdl.M", X1_kdl.M)
print("X1_kdl.p", X1_kdl.p)

separate()
print("LOG")
log_pin = pin.log(X0_pin.actInv(X1_pin))
# log_kdl1 = kdl.diff(X0_kdl.Inverse(), X1_kdl)
# log_kdl2 = kdl.diff(X1_kdl.Inverse(), X0_kdl)
log_kdl1 = kdl.diff(X0_kdl, X1_kdl)
# log_kdl2 = kdl.diff(X1_kdl, X0_kdl)
print("log_pin", log_pin.vector)
print("log_kdl1", log_kdl1)
# print('log_kdl2', log_kdl2)

separate()
print("LINEAR (log not the same, uses position difference)")
p_kdl = X1_kdl.p - X0_kdl.p
print("p_kdl (position difference)\n X1_kdl.p - X0_kdl.p:\n", p_kdl)
print("log_kdl1.vel\n", log_kdl1.vel)
print("log_pin.linear\n", log_pin.linear)

separate()
print("ORIENTATION (equivalent found, it's axis-angle LWA)")
axa_lwa_pin = X0_pin.rotation @ pin.log(X0_pin.rotation.T @ X1_pin.rotation)
print("axa_lwa_pin (axis-angle LWA):")
print("X0_pin.rotation @ pin.log(X0_pin.rotation.T @ X1_pin.rotation)\n", axa_lwa_pin)
print("X0_pin.act(log_pin).angular\n", X0_pin.act(log_pin).angular)
print("log_kdl1.rot\n", log_kdl1.rot)

separate()
separate_shmol()
print("is kdl.Diff a pin.log in LWA ??")
print("orientation coincides")
separate_shmol()
print("log_kdl1\n  v = {}\n  w = {}".format(log_kdl1.vel, log_kdl1.rot))
print("WORLD\nX0_pin.act(log_pin)\n", X0_pin.act(log_pin))
print(
    "LWA\npin.SE3(X0_pin.rotation, p=zeros).act(log_pin)\n",
    pin.SE3(X0_pin.rotation, np.zeros(3)).act(log_pin),
)

# print('what about actInv?')
# separate_shmol()
# print('X0_pin.act(log_pin)\n', X0_pin.actInv(log_pin))
# print('pin.SE3(X0_pin.rotation, p=zeros).actInv(log_pin)\n',
#       pin.SE3(X0_pin.rotation, np.zeros(3)).actInv(log_pin))
