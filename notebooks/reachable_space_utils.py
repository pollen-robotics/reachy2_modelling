# autoflake: skip_file
import itertools

import numpy as np
import numpy.matlib
import pinocchio as pin
import pycapacity
import pycapacity.algorithms as algos
from scipy.spatial import ConvexHull


# direct kinematics functions using pinocchio
def forward(robot, q, frame_name=None):
    """
    Compute the forward kinematics of the robot for the given joint configuration.
    :param robot: The robot model.
    :param q: The joint configuration.
    :param frame_name: The name of the frame for which to compute the forward kinematics.
    :return: The placement of the frame in the world frame.
    """
    data = robot.createData()
    frame_id = robot.getFrameId(frame_name)
    pin.framesForwardKinematics(robot, data, np.array(q))
    return data.oMf[frame_id]


# positon dk
def dk_position(robot, q, frame_name=None):
    """
    Compute the position of the frame in the world frame.
    :param robot: The robot model.
    :param q: The joint configuration.
    :param frame_name: The name of the frame for which to compute the forward kinematics.
    :return: The position of the frame in the world frame.
    """
    return forward(robot, q, frame_name).translation


from CGAL.CGAL_Alpha_wrap_3 import *
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Mesh_3 import *
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3


def alpha_shape_with_cgal(coords, alpha=None):
    """
    Compute the alpha shape of a set of points.
    Retrieved from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

    :param coords : Coordinates of points
    :param alpha: List of alpha values to influence the gooeyness of the border. Smaller numbers don't fall inward as much as larger numbers.
    Too large, and you lose everything!
    :return: Shapely.MultiPolygons which is the hull of the input set of points
    """
    if alpha is None:
        bbox_diag = np.linalg.norm(np.max(coords, 0) - np.min(coords, 0))
        alpha_value = bbox_diag / 5
    else:
        alpha_value = np.mean(alpha)
    # Convert to CGAL point
    points = [Point_3(pt[0], pt[1], pt[2]) for pt in coords]
    # Compute alpha shape
    Q = Polyhedron_3()
    a = alpha_wrap_3(points, alpha_value, 0.01, Q)
    # Q.make_tetrahedron()
    alpha_shape_vertices = np.array(
        [
            (vertex.point().x(), vertex.point().y(), vertex.point().z())
            for vertex in Q.vertices()
        ]
    )
    alpha_shape_faces = np.array(
        [
            np.array(
                [
                    (
                        face.halfedge().vertex().point().x(),
                        face.halfedge().vertex().point().y(),
                        face.halfedge().vertex().point().z(),
                    ),
                    (
                        face.halfedge().next().vertex().point().x(),
                        face.halfedge().next().vertex().point().y(),
                        face.halfedge().next().vertex().point().z(),
                    ),
                    (
                        face.halfedge().next().next().vertex().point().x(),
                        face.halfedge().next().next().vertex().point().y(),
                        face.halfedge().next().next().vertex().point().z(),
                    ),
                    # for i in face.halfedge()
                ]
            )
            for face in Q.facets()
        ]
    )

    return alpha_shape_vertices, alpha_shape_faces


# reachable space calculation algorithm
def curves_reachable_set_dq_nd(
    robot,
    q0,
    time_horizon,
    q_max=None,
    q_min=None,
    dq_max=None,
    dq_min=None,
    options=None,
):
    """
    Compute the reachable set of the robot for the given joint configuration.
    The algorithm calculates the reachable set of cartesian position of the desired frame of the robot given the robots joint position and joint velocity limits.
    The output of the algorithm is the reachable space that the robot is able to reach within the horizon time, while guaranteeing that the joint position and velocity limits are not violated.

    If you are interested in the complete workspace of the robot, you can set a large time horizon (>1 second)

    The parameters of the algorithm are set using the options dictionary. The following options are available:
    - frame_name: The name of the frame for which to compute the reachable set. The default value is None, which means that the reachable set is computed for the origin of the robot.
    - n_samples: The number of samples to use for the discretization of the joint velocity space. The higher the number of samples, the more accurate the reachable set will be, however the longer the computation time will be
    - facet_dim: The dimension of the facet that will be sampled. Between 0 and the number of DOF of the robot.  The higher the number of samples, the more accurate the reachable set will be, however the longer the computation time will be
    - convex: Whether to make the reachable set convex or not. If set to True, the reachable set will be convex, if False the reachable set will be non-convex.

    :param robot: The robot model.
    :param q0: The joint configuration.
    :param time_horizon: The time horizon for which to compute the reachable set.

    :return: The vertices and the faces of the reachable set for the given joint configuration.
    """

    delta_t = time_horizon

    if options is None:
        options = {
            "convex": False,
            "frame": robot.model.frames[-1].name,
            # 'n_samples': 5, # number of samples taken within the horizon (the higher the better approaximation - but the longer execution time)
            # 'facet_dim': 3 # from 1 to number of robot's DOF (the higher the better approximation - but longet the execution time)
        }
        # horizon time
        if delta_t < 0.2:
            options["n_samples"] = 3
            options["facet_dim"] = 1
        elif delta_t < 0.5:
            options["n_samples"] = 3
            options["facet_dim"] = 2
        elif delta_t < 0.9:
            options["n_samples"] = 4
            options["facet_dim"] = 3
        else:
            options["n_samples"] = 4
            options["facet_dim"] = 4

    # get joint position ranges
    if q_max is None:
        q_max = robot.model.upperPositionLimit.T
    if q_min is None:
        q_min = robot.model.lowerPositionLimit.T
    if dq_max is None:
        # get max velocity
        dq_max = robot.model.velocityLimit
    if dq_min is None:
        dq_min = -dq_max

    n_samples = options["n_samples"]
    n_steps = 1
    n_combs = options["facet_dim"]

    if len(dq_min) != len(q_min):
        n = len(dq_max)
        dq_max = np.hstack((dq_max, -np.ones(len(q_min) - n) * 1000))
        dq_min = np.hstack((dq_min, np.ones(len(q_min) - n) * 1000))

    n = len(dq_max)
    n_dof = n_steps * n
    dt = delta_t / n_steps

    x_0 = dk_position(robot.model, q0, options["frame"]).flatten()

    dq_ub = np.matlib.repmat(
        np.minimum(dq_max, (q_max.flatten() - q0) / dt), n_steps, 1
    ).flatten()
    dq_lb = np.matlib.repmat(
        np.maximum(dq_min, (q_min.flatten() - q0) / dt), n_steps, 1
    ).flatten()

    Dq_ub = np.diag(dq_ub)
    Dq_lb = np.diag(dq_lb)
    sum_steps = np.matlib.repmat(np.eye(n), n_steps, 1)

    combs = list(itertools.combinations(range(n_dof), n_combs))
    perm_set = list(itertools.product([1, 0], repeat=n_dof - n_combs))

    dq_curve_v = []

    x_rng = np.arange(0, 1, 1 / n_samples)
    mat_rng = np.array(list(itertools.product(x_rng, repeat=n_combs))).T
    # np.random.seed(1234)
    # mat_rng = np.random.random((n_combs,n_samples*10))
    n_rng = len(mat_rng.T)

    for c in combs:
        c = np.array(c)
        ind = np.ones(n_dof, dtype=bool)
        ind[c] = np.zeros(len(c))
        ind_i = np.argwhere(ind > 0)

        n_ps = len(perm_set)
        ps = np.array(perm_set).T
        dq_i = np.zeros((n_dof, n_ps))
        dq_i[ind, :] = ps

        DQ_i = np.matlib.repmat(dq_i.T, n_rng, 1).T
        DQ_i[ind, :] = (
            DQ_i[ind, :] * Dq_ub[ind_i, ind_i]
            + (1 - DQ_i[ind, :]) * Dq_lb[ind_i, ind_i]
        )
        mat = (
            np.diag([dq_ub[c_i] - dq_lb[c_i] for c_i in c]) @ mat_rng
            + np.array([dq_lb[c_i] for c_i in c])[:, None]
        )
        DQ_i[c, :] = np.matlib.repeat(mat, n_ps, 1)
        dq_curve_v = algos.stack(dq_curve_v, DQ_i.T)

    dq_curve_v = np.unique(dq_curve_v, axis=0)
    q_v = (np.array(q0)[:, None] + (dq_curve_v @ sum_steps).T * dt).T
    x_curves = np.array(
        [dk_position(robot.model, q, options["frame"]).flatten() - x_0 for q in q_v]
    )

    if options["convex"]:
        poly = pycapacity.objects.Polytope(x_curves.T)
        vert = x_curves
        poly.find_faces()
        faces = poly.face_indices
    else:
        if options is not None and "alpha" in options.keys():
            vert, faces = alpha_shape_with_cgal(x_curves, options["alpha"])
        else:
            vert, faces = alpha_shape_with_cgal(x_curves)
        vert = faces.reshape(-1, 3)
        faces = np.arange(len(vert)).reshape(-1, 3)
    return vert, faces
