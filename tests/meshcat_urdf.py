import pinocchio as pin
import numpy as np
import time
import scipy
from copy import copy
from example_robot_data import load

## visualise the robot
from pinocchio.visualize import MeshcatVisualizer

## visualise the polytope and the ellipsoid
import meshcat.geometry as g

# import pycapacity
import pycapacity as pycap

try:
    import reachy2_modelling.old.pin as rp
except ImportError as e:
    print("Error:", e)
    print("is reachy2_modelling installed? run in root directory:")
    print("pip install -e .")
    exit(1)


robot = copy(rp.robot)
# robot.model = rp.model_r_arm
robot.data = robot.model.createData()


# get joint position ranges
# q_max = robot.model.upperPositionLimit.T
# q_min = robot.model.lowerPositionLimit.T
q_min = np.array([-np.pi, -2.9, -np.pi, -2.22, -np.pi / 4, -np.pi / 4, -np.pi])
# q_max = np.array([np.pi, 0.55, np.pi, 0.02,np.pi/4,np.pi/4,np.pi])
q_max = np.array([np.pi, 0, np.pi, 0.0, np.pi / 4, np.pi / 4, np.pi])
robot.model.upperPositionLimit = q_max
robot.model.lowerPositionLimit = q_min
dq_max = np.ones(robot.nq)
dq_min = -dq_max
# get max velocity
t_max = np.ones(robot.nq) * 4  # amps
t_min = -t_max

# Use robot configuration.
# q0 = np.random.uniform(q_min,q_max)
q = (q_min + q_max) / 2

qq = np.zeros(robot.model.nq)
qq[: len(q)] = q


viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
# Start a new MeshCat server and client.
viz.initViewer(open=True)
# Load the robot in the viewer.
viz.loadViewerModel()
viz.display(qq)
# small time window for loading the model
# if meshcat does not visualise the robot properly, augment the time
# it can be removed in most cases
time.sleep(0.2)

viz.viewer.jupyter_cell()
