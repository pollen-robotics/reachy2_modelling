import numpy as np
import pink
import qpsolvers
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector


class PinkIk:
    def __init__(self, pinmodel, q_ref=None):
        # TODO: de-hardcode
        q_min = np.array(
            [-3 * np.pi, -3.7, -3 * np.pi, -2.22, -np.pi / 4, -np.pi / 4, -np.pi]
        )
        q_max = np.array(
            [3 * np.pi, 0.38, 3 * np.pi, 0.02, np.pi / 4, np.pi / 4, np.pi]
        )
        pinmodel.lowerPositionLimit = q_min
        pinmodel.upperPositionLimit = q_max
        pinmodel.velocityLimit = np.ones(pinmodel.nq) * 5

        self.model = pinmodel
        self.data = pinmodel.createData()

        if q_ref is None:
            q_ref = np.random.uniform(q_min, q_max)

        self.end_effector_task = FrameTask(
            "r_arm_tip",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.3,  # [cost] / [rad]
        )

        self.posture_task = PostureTask(
            cost=1e-4,  # [cost] / [rad]
        )
        self.posture_task.set_target(
            np.array([1.57, -1.57, -1.57, -1.57, 0, 0, 0])[: self.model.nq]
        )

        self.tasks = [self.end_effector_task, self.posture_task]

        self.configuration = pink.Configuration(self.model, self.data, q_ref)
        self.end_effector_task.set_target_from_configuration(self.configuration)

        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

    def solve(self, target_pose, dt, q0=None, q_ref=None):

        if q0 is None:
            q0 = self.configuration.q

        if q_ref is not None:
            self.posture_task.set_target(q_ref)

        self.configuration = pink.Configuration(self.model, self.data, q0)

        # Update task targets
        end_effector_target = self.end_effector_task.transform_target_to_world
        end_effector_target.translation = target_pose.translation
        end_effector_target.rotation = target_pose.rotation

        velocity = solve_ik(
            self.configuration, self.tasks, dt, solver=self.solver, damping=1e-7
        )
        self.configuration.integrate_inplace(velocity, dt)

        return self.configuration.q
