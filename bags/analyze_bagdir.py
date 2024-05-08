#!/usr/bin/env python3
import argparse
import code
import csv
import os
import pickle
import sys
import types
from typing import Tuple

import numpy as np
import pandas as pd
import pinocchio as pin
import PyKDL as kdl
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.spatial.transform import Rotation

import reachy2_modelling.pin as rp
import reachy2_modelling.rerun_utils as ru
from reachy2_modelling.old.symik import MySymIK

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
pd.set_option("display.precision", 2)

qcols = [f"q{x+1}" for x in np.arange(7)]
qdcols = [f"qd{x+1}" for x in np.arange(7)]

results = None


def histogram(results):
    ldf = results[results.index.get_level_values("arm") == "l_arm"]
    rdf = results[results.index.get_level_values("arm") == "r_arm"]
    bins = 10
    alpha = 0.6
    bbox_to_anchor = (1.4, 1.00)
    legend_loc = "lower center"
    legend_ncols = 4
    rwidth = 7

    fig1, ((axl1, axr1), (axl2, axr2), (axl3, axr3)) = plt.subplots(3, 2)
    fig1.suptitle("Linear Manipulability")

    def hist_row(col, axl, axr):
        axl.set_title(f"L Arm {col}")
        axr.set_title(f"R Arm {col}")

        def hist(datas, ax):
            labels = [x[0] for x in datas]
            data = [x[1].values for x in datas]
            weights = [np.ones(len(x)) / len(x) for x in data]
            ax.hist(
                data, bins, alpha=alpha, label=labels, weights=weights, rwidth=rwidth
            )
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid()
            # for data in datas:
            # label=data[0]
            # x=data[1].values
            # ax.hist(x, bins, alpha=alpha, label=label, weights=np.ones(len(x)) / len(x), rwidth=rwidth, histtype='barstacked')
            # ax.yaxis.set_major_formatter(PercentFormatter(1))
            # ax.grid()

        datal = ldf.groupby("offset")[col]
        datar = ldf.groupby("offset")[col]
        hist(datal, axl)
        hist(datar, axr)
        # axr.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
        # ldf.groupby("offset").hist(column=col, bins=bins, ax=axl, legend=True, alpha=alpha)
        # rdf.groupby("offset").hist(column=col, bins=bins, ax=axr, legend=True, alpha=alpha)

    hist_row(col="linmanip2", axl=axl1, axr=axr1)
    hist_row(col="linmanip4", axl=axl2, axr=axr2)
    hist_row(col="linmanip7", axl=axl3, axr=axr3)
    fig1.legend(*axr1.get_legend_handles_labels(), loc=legend_loc, ncol=legend_ncols)
    # fig1.tight_layout()

    fig2, ((axl1, axr1), (axl2, axr2), (axl3, axr3)) = plt.subplots(3, 2)
    fig2.suptitle("Full Manipulability")
    hist_row(col="manip2", axl=axl1, axr=axr1)
    hist_row(col="manip4", axl=axl2, axr=axr2)
    hist_row(col="manip7", axl=axl3, axr=axr3)
    fig2.legend(*axr1.get_legend_handles_labels(), loc=legend_loc, ncol=legend_ncols)
    # fig2.tight_layout()


def histogram_qd(results):
    ldf = results[results.index.get_level_values("arm") == "l_arm"]
    rdf = results[results.index.get_level_values("arm") == "r_arm"]
    bins = 20
    alpha = 0.6
    bbox_to_anchor = (1.4, 1.00)
    legend_loc = "lower center"
    legend_ncols = 4
    rwidth = 7

    fig1, axes = plt.subplots(7, 2)
    fig1.suptitle("Joint Velocities")

    for row in range(7):

        def hist_row(col, axl, axr):
            axl.set_title(f"L Arm {col}")
            axr.set_title(f"R Arm {col}")

            def hist(datas, ax):
                labels = [x[0] for x in datas]
                dataseries = [x[1] for x in datas]
                quantmin, quantmax = 0.1, 0.9
                data = [
                    x[x.between(x.quantile(quantmin), x.quantile(quantmax))].values
                    for x in dataseries
                ]
                # data = [x.values for x in dataseries]
                weights = [np.ones(len(x)) / len(x) for x in data]
                ax.hist(
                    data,
                    bins,
                    alpha=alpha,
                    label=labels,
                    weights=weights,
                    rwidth=rwidth,
                )
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid()

            datal = ldf.groupby("offset")[col]
            datar = ldf.groupby("offset")[col]
            hist(datal, axl)
            hist(datar, axr)

        hist_row(col=f"qd{row+1}", axl=axes[row][0], axr=axes[row][1])
    fig1.legend(
        *axes[0][0].get_legend_handles_labels(), loc=legend_loc, ncol=legend_ncols
    )
    # fig1.tight_layout()


def save_result(arm_name, offset_name, df):
    if "arm" not in df:
        df["arm"] = arm.name
    if "offset" not in df:
        df["offset"] = offset_name
    df = df.set_index(["arm", "offset"], append=True)
    global results
    if results is None:
        results = df
    else:
        results = pd.concat([results, df], sort=True)

    # if arm_name not in results:
    #     results[arm_name] = {}
    # results[arm_name][offset_name] = df


def separate_big():
    print(70 * "_")
    print(70 * "_")
    print(70 * "_")


def separate():
    print(40 * "-")
    print(40 * "-")


def pklpath(bagdir, offset, arm_name):
    return os.path.join(bagdir, f"{arm_name}_target_pose_{offset}.pkl")


def load_pkl(bagdir, offset, arm_name):
    pkl_fname = pklpath(bagdir, offset, arm_name)

    df = None
    if os.path.isfile(pkl_fname):
        print("pkl file found, loading: {}".format(pkl_fname))
        return pd.read_pickle(pkl_fname)
    else:
        print("pkl file NOT found, processing ({})".format(pkl_fname))

    return None


def manip_factory(arm, tip, name, njoints, linear):
    func = arm.manip
    if linear:
        func = arm.linmanip

    def series_to_manip(series):
        q = series[qcols].values.astype(float)
        data = {}
        data[name] = func(q, tip, njoints=dof)
        return pd.Series(data=data)

    return series_to_manip


def series_to_ik_factory(arm, ik):
    def series_to_ik(series):
        M = series_to_mat(series)
        q, reachable, multiturn = arm.ik(ik, M)

        data = {}
        for i, ang in enumerate(q):
            data[qcols[i]] = ang
        data["reachable"] = reachable
        data["multiturn"] = multiturn

        return pd.Series(data=data)

    return series_to_ik


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
parser.add_argument("bagdir", type=str)
parser.add_argument(
    "--force", action="store_true", help="force recomputing even if pkls are found"
)
parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="enter interactive python session after loading files",
)
args = parser.parse_args()

larmf = os.path.join(args.bagdir, "l_arm_target_pose.csv")
rarmf = os.path.join(args.bagdir, "r_arm_target_pose.csv")
print("load csv files:", larmf, rarmf)
csv_ldf, csv_rdf = ru.df_target_poses(larmf, rarmf)
ru.df_add_relative_time(csv_ldf)
ru.df_add_relative_time(csv_rdf)


def index_todatetime(df):
    df = df.index = pd.to_datetime(df.index, unit="s")


index_todatetime(csv_ldf)
index_todatetime(csv_rdf)

offsets = [
    ([10, 0, 15], "beta"),  # current: config 0
    ([0, 0, 0], "straight"),
    ([10, 0, 15], "backwards"),
    ([10, 0, 15], "upwards"),
]

for offset in offsets:
    ik = MySymIK()
    offset_name = offset[1]
    separate_big()
    print(f"processing offset named {offset_name}...")

    roll, pitch, yaw = [*offset[0]]
    print("RPY:", roll, pitch, yaw)
    urdf_str, urdf_path = rp.offset_urdf(roll, pitch, yaw)
    models = rp.Models(urdf_str)
    larm = ru.ArmHandler("l_arm", models.l_arm)
    rarm = ru.ArmHandler("r_arm", models.r_arm)

    def load_or_copy(arm, csvdf, force_processing):
        needs_processing = False
        df = None
        if not force_processing:
            df = load_pkl(args.bagdir, offset_name, arm.name)
        else:
            print("--force provided, will not try to reload pickle files")
        if df is None:
            df = csvdf.copy(deep=True)
            needs_processing = True
        return df, needs_processing

    ldf, lneeds_processing = load_or_copy(larm, csv_ldf, args.force)
    rdf, rneeds_processing = load_or_copy(rarm, csv_rdf, args.force)

    for needs_processing, arm, df in zip(
        [lneeds_processing, rneeds_processing], [larm, rarm], [ldf, rdf]
    ):
        separate()
        if not needs_processing:
            print(f"skipping {arm.name} (loaded pkl)")
            save_result(arm.name, offset_name, df)
            continue

        print(f"processing {arm.name}...")

        print("computing ik...")
        df = pd.concat([df, df.apply(series_to_ik_factory(arm, ik), axis=1)], axis=1)

        print("computing qdot...")
        df = pd.concat([df, df.apply(series_to_qd, axis=1)], axis=1)

        print("computing manipulability...")
        dofs = [2, 4, 7]
        for dof in dofs:
            print(f"- dof:{dof}")
            if dof != 7:
                tip = arm.njoint_name(dof)
            else:
                tip = arm.tip
            df = pd.concat(
                [
                    df,
                    df.apply(
                        manip_factory(
                            arm,
                            tip=tip,
                            name=f"linmanip{dof}",
                            njoints=dof,
                            linear=True,
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
                            arm, tip=tip, name=f"manip{dof}", njoints=dof, linear=False
                        ),
                        axis=1,
                    ),
                ],
                axis=1,
            )

        pkl_fname = pklpath(args.bagdir, offset_name, arm.name)
        print(f"saving {pkl_fname}...")
        df.to_pickle(pkl_fname)
        save_result(arm.name, offset_name, df)

histogram(results)
histogram_qd(results)
plt.show()

if args.interactive:
    code.interact(local=dict(globals(), **locals()))
