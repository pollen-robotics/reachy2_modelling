import numpy as np
import pinocchio as pin
import PyKDL as kdl

try:
    import reachy2_modelling.old.kdl as rk
    import reachy2_modelling.old.pin as rp
    import reachy2_modelling.old.rtb as rr
except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

USE_KDL_WORLD_FRAME = False
USE_KDL_WORLD_FRAME = True


def separate(more=False):
    if more:
        print("*" * 60)
        print("*" * 60)
    else:
        print("-" * 30)


def kdlFrame_to_np(kdlframe):
    mat = np.zeros((4, 4))
    for i in range(3):
        mat[i, 3] = kdlframe.p[i]
    for i in range(3):
        for j in range(3):
            mat[i, j] = kdlframe[i, j]
    return mat


def kdlF_to_SE3(X):
    return pin.SE3(kdlFrame_to_np(X))


def pin_FK(model, data, q, tip, head=False, world_frame=False):
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    frame_id = model.getFrameId(tip)
    X = data.oMf[frame_id]

    # if not world, then it's torso
    if not world_frame:
        X = data.oMf[model.getFrameId("torso")].actInv(X)
    return X


def rtb_FK(robot, q, tip, world_frame=False):

    if world_frame:
        X = pin.SE3(robot.fkine(q0, end=tip).A)
    else:
        Xworld_torso = robot.fkine(q0, end="torso")
        Xworld_ee = robot.fkine(q0)
        X = pin.SE3((Xworld_torso.inv() * Xworld_ee).A)
        X = pin.SE3((robot.fkine(q0, start="torso", end=tip)).A)
    return X


def kdl_FK(fk_solver, q):
    qq = kdl.JntArray(len(q))
    for i, j in enumerate(q):
        qq[i] = j

    pose = kdl.Frame()
    _ = fk_solver.JntToCart(qq, pose)
    return pose


def fk_all(fk_solver, model, data, robot, q0, tip, kdl_world_frame):

    separate(True)
    print("TIP:", tip)
    separate()
    arm_flag = "r_" in tip or "l_" in tip
    Xkdl = kdlF_to_SE3(kdl_FK(fk_solver, q0))
    Xpin = pin_FK(model, data, q0, tip, head=not arm_flag, world_frame=kdl_world_frame)
    Xrtb = rtb_FK(robot, q0, tip, kdl_world_frame)
    print("Xkdl", Xkdl)
    X = Xkdl
    print(pin.rpy.matrixToRpy(X.rotation))
    print("Xpin", Xpin)
    X = Xpin
    print(pin.rpy.matrixToRpy(X.rotation))
    print("Xrtb", Xrtb)
    X = Xrtb
    print(pin.rpy.matrixToRpy(X.rotation))

    separate()
    print("KDL vs PIN")

    def errs(pdiff, logR):
        print(
            "   Position err        ({:.4f}) : {}".format(np.linalg.norm(pdiff), pdiff)
        )
        print("Orientation err (logR) ({:.4f}) : {}".format(np.linalg.norm(logR), logR))

    pdiff = Xpin.translation - Xkdl.translation
    R = Xpin.rotation.T @ Xkdl.rotation
    logR = pin.log(R)
    errs(pdiff, logR)
    # print('Rpin.T@Rkdl', R)
    # print(pin.rpy.matrixToRpy(R))

    separate()
    print("KDL vs RTB")

    pdiff = Xrtb.translation - Xkdl.translation
    logR = pin.log(Xrtb.rotation.T @ Xkdl.rotation)
    errs(pdiff, logR)

    return Xkdl, Xpin, Xrtb


#################################
if USE_KDL_WORLD_FRAME:
    fk_solver, kdl_world_frame = rk.fk_solver_l_arm, rk.world_frame(rk.chain_l_arm)
else:
    fk_solver, kdl_world_frame = rk.fk_solver_l_arm_torso, rk.world_frame(
        rk.chain_l_arm_torso
    )
tip = "l_arm_tip"
model, robot = rp.model_l_arm, rr.robot_l_arm
frames_names = [x.name for x in model.frames.tolist()]
data = model.createData()
q0 = np.random.rand(7)
Xkdl, Xpin, Xrtb = fk_all(
    fk_solver, model, data, robot, q0, tip, kdl_world_frame=kdl_world_frame
)

#################################
tip = "r_arm_tip"
if USE_KDL_WORLD_FRAME:
    fk_solver, kdl_world_frame = rk.fk_solver_r_arm, rk.world_frame(rk.chain_r_arm)
else:
    fk_solver, kdl_world_frame = rk.fk_solver_r_arm_torso, rk.world_frame(
        rk.chain_r_arm_torso
    )
model, robot = rp.model_r_arm, rr.robot_r_arm
data = model.createData()
fk_all(fk_solver, model, data, robot, q0, tip, kdl_world_frame=kdl_world_frame)

#################################
tip = "head_tip"
if USE_KDL_WORLD_FRAME:
    fk_solver, kdl_world_frame = rk.fk_solver_head, rk.world_frame(rk.chain_head)
else:
    fk_solver, kdl_world_frame = rk.fk_solver_head_torso, rk.world_frame(
        rk.chain_head_torso
    )
model, robot = rp.model_head, rr.robot_head
data = model.createData()
fk_all(fk_solver, model, data, robot, q0[:3], tip, kdl_world_frame=kdl_world_frame)
