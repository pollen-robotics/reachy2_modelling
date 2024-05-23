#!/usr/bin/env python3
import argparse
import code
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
import reachy2_modelling as r2
import reachy2_modelling.rerun_utils as ru
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.spatial.transform import Rotation

print("matplotlibrc:", matplotlib.matplotlib_fname())
plt.style.use("classic")


SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", titleweight="roman")  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
# 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold'
plt.rc("figure", titleweight="bold")

# # plt.rc('legend',fontsize=10) # using a size in points
# plt.rcParams.update({'font.size': 10})
# plt.rc('legend',fontsize='medium') # using a named size

plt.rcParams["figure.constrained_layout.use"] = True

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
pd.set_option("display.precision", 2)

qcols = [f"q{x+1}" for x in np.arange(7)]
qdcols = [f"qd{x+1}" for x in np.arange(7)]

# where final results will be stored
results = None

# used for computing the manipulability
manip_dofs = [2, 4, 7]

# the order to show the labels in graphs
ordered_labels = [
    "beta",
    "straight",
    "backwards",
    "upwards-5",
    "upwards-10",
    "upwards-20",
    "backo-upwards",
]


@dataclass
class Result:
    name: str
    offset: str
    df: pd.DataFrame


def arm_graphs(df, arm_name, no_joints, images, bagdir):
    bins = 8
    alpha = 0.6
    bbox_to_anchor = None
    legend_loc = "upper right"

    legend_loc = "center"
    bbox_to_anchor = (0.7, 0.99)
    # legend_loc = "lower center"
    legend_ncols = 10
    rwidth = 7
    figsize = (20, 11)
    dpi = 300
    title_xloc = 0.2

    subplot_cols = 5 - no_joints * 2

    fig1, axes1 = plt.subplots(4, subplot_cols)
    axes1 = np.array(axes1)
    fig1.suptitle(f"{bagdir}: {arm_name} joints 1,2,3,4", x=title_xloc)

    # q1 qd1 hist_qd1 manip2 linmanip2
    # q2 qd2 hist_qd2 manip3 linmanip3
    # q3 qd3 hist_qd3 manip4 linmanip4
    # q4 qd4 hist_qd4 reach  multiturn

    def data_to_plot(col):
        datas = df.groupby("offset")[col]
        labels = [x[0] for x in datas]
        dataseries = [x[1] for x in datas]
        global ordered_labels
        ordered_dataseries = []
        for label in ordered_labels:
            idx = labels.index(label)
            ordered_dataseries.append(dataseries[idx])

        assert len(dataseries) == len(
            ordered_dataseries
        ), "check if missing ordered_labels from the desired offset list"
        " or maybe need to --force a recompute"
        assert len(labels) == len(ordered_labels)
        return ordered_dataseries, ordered_labels

    def plot(ax, dataseries, labels, title, xlabel=None, ylabel=None):
        ax.set_title(title)
        datas = [x.values for x in dataseries]
        times = [x[1].values for x in df.groupby("offset")["time"]]

        ax.plot(
            np.array(times).T,
            np.array(datas).T,
            alpha=alpha,
            label=labels,
        )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid()

    def hist_qd(ax, dataseries, labels, title, quantmin=0.15, quantmax=0.85):
        hist(
            ax,
            dataseries,
            labels,
            title,
            xlabel="rad/s",
            quantmin=quantmin,
            quantmax=quantmax,
        )

    def hist(ax, dataseries, labels, title, xlabel=None, quantmin=0, quantmax=1):
        ax.set_title(title)
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
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.grid()

    for row in range(4):
        xlabel = "[s]"
        if not no_joints:
            var, ylabel = f"q{row+1}", "rad"
            plot(axes1[row, 0], *data_to_plot(var), var, xlabel, ylabel)
            var, ylabel = f"qd{row+1}", "rad/s"
            plot(axes1[row, 1], *data_to_plot(var), var, xlabel, ylabel)
        var = f"qd{row+1}"
        hist_qd(axes1[row, 2 - no_joints * 2], *data_to_plot(var), var)

    for row, dof in enumerate(manip_dofs):
        var = f"linmanip{dof}"
        hist(axes1[row, 3 - no_joints * 2], *data_to_plot(var), var)

        var = f"manip{dof}"
        hist(axes1[row, 4 - no_joints * 2], *data_to_plot(var), var)

    var = "reachable"
    plot(axes1[3, 3 - no_joints * 2], *data_to_plot(var), var)
    var = "multiturn"
    plot(axes1[3, 4 - no_joints * 2], *data_to_plot(var), var)

    fig1.legend(
        *axes1[0, 3].get_legend_handles_labels(),
        loc=legend_loc,
        ncol=legend_ncols,
        bbox_to_anchor=bbox_to_anchor,
        # prop={'width':5},
    )
    # fig1.tight_layout()
    if images:
        img_file = f"{bagdir}/{bagdir}_{arm_name}_joints1234.png"
        print(f"saving image file {img_file}...")
        fig1.set_size_inches(*figsize)
        fig1.savefig(img_file, bbox_inches="tight", dpi=dpi)

    fig2, axes2 = plt.subplots(4, subplot_cols)
    fig2.suptitle(f"{bagdir}: {arm_name} wrist joints (5,6,7)", x=title_xloc)
    for row in range(3):
        xlabel = "s"
        if not no_joints:
            var, ylabel = f"q{row+1+3}", "rad"
            plot(axes2[row, 0], *data_to_plot(var), var, xlabel, ylabel)
            var, ylabel = f"qd{row+1}", "rad/s"
            plot(axes2[row, 1], *data_to_plot(var), var, xlabel, ylabel)

        var = f"qd{row+1}"
        hist_qd(axes2[row, 2 - no_joints * 2], *data_to_plot(var), var)

    for row, dof in enumerate(manip_dofs):
        var = f"linmanip{dof}"
        hist(axes2[row, 3 - no_joints * 2], *data_to_plot(var), var)

        var = f"manip{dof}"
        hist(axes2[row, 4 - no_joints * 2], *data_to_plot(var), var)

    var = "reachable"
    plot(axes2[3, 3 - no_joints * 2], *data_to_plot(var), var)
    var = "multiturn"
    plot(axes2[3, 4 - no_joints * 2], *data_to_plot(var), var)

    fig2.legend(
        *axes1[0, 3].get_legend_handles_labels(),
        loc=legend_loc,
        ncol=legend_ncols,
        bbox_to_anchor=bbox_to_anchor,
    )
    if images:
        img_file = f"{bagdir}/{bagdir}_{arm_name}_joints567.png"
        print(f"saving image file {img_file}...")
        fig2.set_size_inches(*figsize)
        fig2.savefig(img_file, bbox_inches="tight", dpi=dpi)


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
        datar = rdf.groupby("offset")[col]
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
            datar = rdf.groupby("offset")[col]
            hist(datal, axl)
            hist(datar, axr)

        hist_row(col=f"qd{row+1}", axl=axes[row][0], axr=axes[row][1])
    fig1.legend(
        *axes[0][0].get_legend_handles_labels(), loc=legend_loc, ncol=legend_ncols
    )
    # fig1.tight_layout()


def save_result(res):
    arm_name = res.name
    offset_name = res.offset
    df = res.df
    if "arm" not in df:
        df["arm"] = arm_name
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
        data[name] = func(q, tip, njoints=njoints)
        return pd.Series(data=data)

    return series_to_manip


def series_to_ik_factory(symarm):
    def series_to_ik(series):
        M = series_to_mat(series)
        q, reachable, multiturn = symarm.ik(M)

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
parser.add_argument("--images", action="store_true", help="generate image files")
parser.add_argument(
    "-j",
    help="number of threads to use",
    default=mp.cpu_count(),
)
parser.add_argument(
    "--no-joints",
    dest="no_joints",
    action="store_true",
    help="don't show joint q and qd in graphs",
)
parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="enter interactive python session after loading files",
)
args = parser.parse_args()

print(f"gonna use {args.j} threads (change with -jN)")

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
    ([0, 0, -5], "backwards"),
    ([-5, 0, 0], "upwards-5"),
    ([-10, 0, 0], "upwards-10"),
    ([-20, 0, 0], "upwards-20"),
    ([-10, 0, -5], "backo-upwards"),
]

assert len(offsets) == len(ordered_labels), (
    "var ordered_labels should contain all the requested offset names\n"
    f"  offset names: {[x[1] for x in offsets]}\n"
    f"ordered_labels: {[x for x in ordered_labels]}\n"
)


def process_offset(params):
    offset_params, arm_name = params[0], params[1]
    offset_name = offset_params[1]
    offset_rpy = offset_params[0]

    df = None
    if args.force:
        print(
            f"--force used (no pickle file reload), recompute: {arm_name}:{offset_name}"
        )
    else:
        df = load_pkl(args.bagdir, offset_name, arm_name)

    # if df loaded, finish, else process
    if df is not None:
        return Result(name=arm_name, offset=offset_name, df=df)
    csvdf = csv_rdf
    if arm_name == "l_arm":
        csvdf = csv_ldf

    # don't overwrite original df
    df = csvdf.copy(deep=True)

    symarm = r2.symik.SymArm(arm_name, shoulder_offset=offset_rpy)
    pinwrapper = r2.pin.PinWrapperArm.from_shoulder_offset(arm_name, *offset_rpy)

    total = 4 + len(manip_dofs)

    def log(msg, inc_stage=True):
        global stage
        base = f"{arm_name}::{offset_name}::{msg}"
        print(f"{base:<40} |  phase:{log.stage}/{total}")
        if inc_stage:
            log.stage += 1

    log.stage = 0

    roll, pitch, yaw = [*offset_rpy]
    log(f"RPY:{roll},{pitch},{yaw}")

    log("ik")
    df = pd.concat([df, df.apply(series_to_ik_factory(symarm), axis=1)], axis=1)

    log("qdot")
    df = pd.concat([df, df.apply(series_to_qd, axis=1)], axis=1)

    log("manip")
    for dof in manip_dofs:
        log(f"manip (dof:{dof})")
        if dof != 7:
            tip = pinwrapper.njoint_name(dof)
        else:
            tip = pinwrapper.tip
        df = pd.concat(
            [
                df,
                df.apply(
                    manip_factory(
                        pinwrapper,
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
                        pinwrapper,
                        tip=tip,
                        name=f"manip{dof}",
                        njoints=dof,
                        linear=False,
                    ),
                    axis=1,
                ),
            ],
            axis=1,
        )

    pkl_fname = pklpath(args.bagdir, offset_name, arm_name)
    log(f"save pkl file")
    df.to_pickle(pkl_fname)

    return Result(name=arm_name, offset=offset_name, df=df)


parameters = [
    (offset_params, arm_name)
    for offset_params, arm_name in zip(offsets, ["l_arm" for _ in offsets])
]
parameters += [
    (offset_params, arm_name)
    for offset_params, arm_name in zip(offsets, ["r_arm" for _ in offsets])
]


current_results = []
start = time.time()

with mp.Pool(args.j) as p:
    current_results = p.map(process_offset, parameters)
print("done!")

for res in current_results:
    save_result(res)


print(f"Total time: {time.time()-start:.2f}s")

ldf = results[results.index.get_level_values("arm") == "l_arm"]
rdf = results[results.index.get_level_values("arm") == "r_arm"]
# histogram(results)
# histogram_qd(results)
arm_graphs(
    ldf, "l_arm", no_joints=args.no_joints, images=args.images, bagdir=args.bagdir
)
arm_graphs(
    rdf, "r_arm", no_joints=args.no_joints, images=args.images, bagdir=args.bagdir
)
if not args.images:
    plt.show()
else:
    print("--images provided, image files were saved and will not be shown")


if args.interactive:
    code.interact(local=dict(globals(), **locals()))
