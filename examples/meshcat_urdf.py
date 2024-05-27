import code

## visualise the polytope and the ellipsoid
import numpy as np

## visualise the robot
from pinocchio.visualize import MeshcatVisualizer

try:
    import reachy2_modelling as r2
except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)


arm = r2.pin.r_arm
model, data = arm.modeldata()


# get joint position ranges
# q_max = model.upperPositionLimit.T
# q_min = model.lowerPositionLimit.T
q_min = np.array([-np.pi, -2.9, -np.pi, -2.22, -np.pi / 4, -np.pi / 4, -np.pi])
# q_max = np.array([np.pi, 0.55, np.pi, 0.02,np.pi/4,np.pi/4,np.pi])
q_max = np.array([np.pi, 0, np.pi, 0.0, np.pi / 4, np.pi / 4, np.pi])
model.upperPositionLimit = q_max
model.lowerPositionLimit = q_min
dq_max = np.ones(model.nq)
dq_min = -dq_max
# get max velocity
t_max = np.ones(model.nq) * 4  # amps
t_min = -t_max

# Use robot configuration.
# q0 = np.random.uniform(q_min,q_max)
q = (q_min + q_max) / 2

qq = np.zeros(model.nq)
qq[: len(q)] = q

# TODO: visual model has some artifacts
# viz = MeshcatVisualizer(model, arm.model_collision(), arm.model_visual())
viz = MeshcatVisualizer(model, arm.model_collision(), arm.model_collision())
# Start a new MeshCat server and client.
viz.initViewer(open=True)
# Load the robot in the viewer.
viz.loadViewerModel()
viz.display(qq)

# small time window for loading the model
# if meshcat does not visualise the robot properly, augment the time
# it can be removed in most cases
# time.sleep(1)
print("control+click to open meshcat url above")

code.interact(local=dict(globals(), **locals()))
