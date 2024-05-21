#!/usr/bin/env python3
import argparse
import code
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

pd.options.plotting.backend = "plotly"

# from reachy2_modelling.old.symik import (
#     MySymIK,
#     data_larm,
#     data_rarm,
#     model_larm,
#     model_rarm,
#     fk, jacobian_joint, jacobian_frame, manip, svals,
# )
from reachy2_modelling.old.symik import *

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

qcols = [f"q{x+1}" for x in np.arange(7)]
qdcols = [f"qd{x+1}" for x in np.arange(7)]


def print_all(dff):
    for index_series in dff.iterrows():
        (epoch_s, series) = index_series
        print(epoch_s, series.values)


def manip_factory(modeldata, tip, name):
    def series_to_linmanip(series):
        q = series[qcols].values.astype(float)
        J = jacobian_joint(q, modeldata=modeldata, tip=tip)[:3, :]
        data = {}
        data[name] = manip(J)
        return pd.Series(data=data)

    return series_to_linmanip


def series_to_ik(series):
    M = series_to_mat(series)
    q, reachable, multiturn = ik.symbolic_inverse_kinematics(arm_str, M)

    data = {}
    for i, ang in enumerate(q):
        data[qcols[i]] = ang
    data["reachable"] = reachable
    data["multiturn"] = multiturn

    return pd.Series(data=data)


def series_to_qd(series):
    q = series[qcols].values.astype(float)
    time = series.name
    if series_to_qd.prev_q is not None:
        dt = time - series_to_qd.prev_time_s
        qd = (q - series_to_qd.prev_q) / dt
    else:
        qd = np.zeros_like(q)

    series_to_qd.prev_q = q
    series_to_qd.prev_time_s = time

    data = {}
    for i, vel in enumerate(qd):
        data[qdcols[i]] = vel

    return pd.Series(data=data)


series_to_qd.prev_q = None
series_to_qd.prev_time_s = None


def series_to_mat(series):
    target_pose_cols = ["pos_x", "pos_y", "pos_z", "or_x", "or_y", "or_z", "or_w"]

    def coords_array_to_mat(coords):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(coords[3:]).as_matrix()
        M[:3, 3] = coords[:3]
        return M

    return coords_array_to_mat(series[target_pose_cols].values)


parser = argparse.ArgumentParser()
parser.add_argument("csvfile", type=str)
args = parser.parse_args()

print("csvfile:", args.csvfile)
csvfilebase = args.csvfile[:-4]
datalist = []
pkl_fname = "{}.pkl".format(csvfilebase)
df = None
if os.path.isfile(pkl_fname):
    print("pkl file found, loading: {}".format(pkl_fname))
    print("run with python -i or modify scrtipt to use the data")
    df = pd.read_pickle(pkl_fname)
else:
    print("pkl file NOT ({}) found, processing".format(pkl_fname))

ik = MySymIK()
if "l_arm" not in csvfilebase and "r_arm" not in csvfilebase:
    print(
        'Error: {} does not contain "l_arm" or "l_arm" in name to determine which model'.format(
            csvfilebase
        )
    )
    sys.exit(1)

# tip = 'l_elbow_ball_link' # as frame
# tip = 'l_elbow_dummy_link'
# tip = 'l_shoulder_ball_link'
tip = "l_elbow_pitch"  # joint
# tip = None

arm_str = "l_arm"
arm_model = model_larm
arm_data = data_larm
if "r_arm" in csvfilebase:
    arm_str = "r_arm"
    arm_model = model_rarm
    arm_data = data_rarm


def correct_arm(ttip):
    if "r_arm" in csvfilebase:
        tip[:3].replace("l_", "r_")
    else:
        tip[:3].replace("r_", "l_")
    return ttip


print("load csv as dataframe...")
if df is None:
    df = pd.read_csv(args.csvfile)
    df = df.set_index("epoch_s", verify_integrity=True)
    df["time"] = df.index - df.index[0]
    df = df.set_index("time", verify_integrity=True)
    # df = df.head(1000) # debug

    df = pd.concat([df, df.apply(series_to_ik, axis=1)], axis=1)
    df = pd.concat([df, df.apply(series_to_qd, axis=1)], axis=1)

    modeldata = (arm_model, arm_data)
    df = pd.concat(
        [
            df,
            df.apply(
                manip_factory(
                    modeldata, tip=correct_arm("l_elbow_pitch"), name="linmanip2"
                ),
                axis=1,
            ),
        ],
        axis=1,
    )
    df = pd.concat(
        [
            df,
            df.apply(
                manip_factory(
                    modeldata, tip=correct_arm("l_elbow_yaw"), name="linmanip4"
                ),
                axis=1,
            ),
        ],
        axis=1,
    )

    # iteration example
    # for index_series in df.iterrows():
    #     (epoch_s, series) = index_series
    #     M = series_to_mat(series)
    #     q, reachable, multiturn = ik.symbolic_inverse_kinematics(arm_str, M)

    #     fkk = fk(q, modeldata=(arm_model, arm_data), tip=tip)
    #     # J = jacobian_frame(q, tip=tip)[:3, :]
    #     J = jacobian_joint(q, modeldata=(arm_model, arm_data), tip=tip)[:3, :]
    #     manipp = manip(J)  # TODO: NOT WORKING
    #     rank = np.linalg.matrix_rank(J)
    #     svalues = svals(J)
    #     nsvalues = 2

    print("saving results to pkl file ({})".format(pkl_fname))
    df.to_pickle(pkl_fname)


figures = [
    px.line(df, y=qcols, title="joint angles"),
    px.line(df, y=qdcols, title="joint velocities"),
]

fig = make_subplots(rows=len(figures), cols=1)

for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.append_trace(figure["data"][trace], row=i + 1, col=1)

plot(fig)
fig.show()
code.interact(local=locals())
