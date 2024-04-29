#!/usr/bin/env python3
import os
import argparse
import subprocess


def flines(fname):
    return sum(1 for _ in open(fname))


def blank(fname):
    with open(fname, "w") as f:
        f.write("")


parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
args = parser.parse_args()
print("bagdir:", args.bagdir)


larmf = os.path.join(args.bagdir, "l_arm_target_pose.csv")
rarmf = os.path.join(args.bagdir, "r_arm_target_pose.csv")
blank(larmf)
blank(rarmf)
print("CSVS:", larmf, rarmf)

lcmd = f"./topic_to_csv.py /l_arm/target_pose {larmf}"
rcmd = f"./topic_to_csv.py /r_arm/target_pose {rarmf}"
lproc = subprocess.Popen(lcmd, shell=True)
rproc = subprocess.Popen(rcmd, shell=True)

print("wait for subscriptions...")
while (flines(larmf) == 0) or (flines(rarmf) == 0):
    pass

print("play ros bag...")
cmd = f"ros2 bag play -r 100 {args.bagdir}"
subprocess.check_output(cmd, shell=True)


# header:
# stamp:
# sec: 1709043931
# nanosec: 376224964
# frame_id: ''
# pose:
# position:
# x: 0.3464586138725281
# y: 0.17137935757637024
# z: -0.287570595741272
# orientation:
# x: 0.03157906025858677
# y: 0.709536301943057
# z: 0.18047085180472908
# w: -0.6804346190686003

# header="stamp_sec,stamp_nanosec,stamp_frame_id,pos_x,pos_y,pos_z,or_x,or_y,or_z,or_w\n"
# sed -i "1s/^/${header}/" ${larmf}
# sed -i "1s/^/${header}/" ${rarmf}

print("count lines...")
cmd = f"wc -l {larmf} {rarmf}"
print(subprocess.check_output(cmd, shell=True).rstrip().decode("utf-8"))
