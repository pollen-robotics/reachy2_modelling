#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def coords_array_to_mat(coords):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(coords[3:]).as_matrix()
    M[:3, 3] = coords[:3]
    return M


np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
args = parser.parse_args()
print("bagdir:", args.bagdir)


larmf = os.path.join(args.bagdir, "l_arm_target_pose.csv")
rarmf = os.path.join(args.bagdir, "r_arm_target_pose.csv")
error = False
for armf in [larmf, rarmf]:
    armf_found = os.path.isfile(armf)
    if not armf_found:
        print(f"Error: {armf} not found")
        error = True
if error:
    exit(1)

print("CSVS:", larmf, rarmf)


dfl = pd.read_csv(larmf)
dfl = dfl.set_index("epoch_s", verify_integrity=True)
dfr = pd.read_csv(rarmf)
dfr = dfr.set_index("epoch_s", verify_integrity=True)

dflr = dfl.join(dfr, lsuffix="_left", rsuffix="_right", how="outer")


target_pose_cols = ["pos_x", "pos_y", "pos_z", "or_x", "or_y", "or_z", "or_w"]
lcols = [x + "_left" for x in target_pose_cols]
rcols = [x + "_right" for x in target_pose_cols]
first = None
for i, index_series in enumerate(dflr.iterrows()):
    if first is None:
        first = index_series
    (index, series) = index_series
    print("i", i, "index", index)

    larm_series = series[lcols]
    print("l", larm_series.values)
    hasnan = larm_series.isnull().values.any()
    print("has nan:", hasnan)
    if not hasnan:
        print(coords_array_to_mat(larm_series.values))

    rarm_series = series[rcols]
    print("r", rarm_series.values)
    hasnan = rarm_series.isnull().values.any()
    print("has nan:", hasnan)
    if not hasnan:
        print(coords_array_to_mat(rarm_series.values))
    if i > 10:
        exit(0)
    print(10 * "-")
