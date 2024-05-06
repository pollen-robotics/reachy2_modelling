#!/usr/bin/env python3
import argparse
import os
import subprocess


def flines(fname):
    return sum(1 for _ in open(fname))


def blank(fname):
    with open(fname, "w") as f:
        f.write("")


parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
parser.add_argument(
    "--verify", action="store_true", help="only verifies csv files are correct"
)
args = parser.parse_args()
print("bagdir:", args.bagdir)


larmf = os.path.join(args.bagdir, "l_arm_target_pose.csv")
rarmf = os.path.join(args.bagdir, "r_arm_target_pose.csv")

if args.verify:
    print("--verify provided, will only check csv files (and not play bag)")
else:
    print("blank csv files:", larmf, rarmf)
    blank(larmf)
    blank(rarmf)
    lcmd = f"./topic_to_csv.py /l_arm/target_pose {larmf}"
    rcmd = f"./topic_to_csv.py /r_arm/target_pose {rarmf}"
    lproc = subprocess.Popen("exec " + lcmd, shell=True)
    rproc = subprocess.Popen("exec " + rcmd, shell=True)

    print("wait for subscriptions...")
    while (flines(larmf) == 0) or (flines(rarmf) == 0):
        pass

    replay_logfile = f"{args.bagdir.rstrip('/')}.log"
    print(f"ros bag replay log: {replay_logfile}")
    print("play ros bag...")
    cmd = f"ros2 bag play --qos-profile-overrides-path reliability_override.yaml -r 30 {args.bagdir} > {replay_logfile} 2>&1"
    subprocess.check_output(cmd, shell=True)

    lproc.kill()
    rproc.kill()


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

# print("count lines...")
# cmd = f"wc -l {larmf} {rarmf}"
# print(subprocess.check_output(cmd, shell=True).rstrip().decode("utf-8"))

sqlcmd = "SELECT topics.id, topics.name, count(messages.topic_id) from topics, messages where messages.topic_id=topics.id group by topics.name;"
try:
    subprocess.check_output(["which", "sqlite3"])
except subprocess.CalledProcessError:
    print("sqlite3 not found, can't verify rosbag size")
    if args.verify:
        exit(1)
    exit(0)

print(25 * "-")
dirr = args.bagdir
dbs = [
    fname
    for fname in os.listdir(dirr)
    if os.path.isfile(os.path.join(dirr, fname)) and fname.endswith(".db3")
]
if len(dbs) > 1:
    print("Error: multiple dbs found in bagdir?:", dbs)
    exit(1)

cmd = f"sqlite3 {args.bagdir}/{dbs[0]} '{sqlcmd}'"
# print(cmd)
rawstr = subprocess.check_output(cmd, shell=True).strip().decode("utf-8")
larm_inline = ["l_arm" in line for line in rawstr.split("\n")]
linecount_pertopic = [int(line.split("|")[-1]) for line in rawstr.split("\n")]
assert len(larm_inline) == len(linecount_pertopic)

count_correct = True
for larm_flag, count in zip(larm_inline, linecount_pertopic):
    armf = larmf if larm_flag else rarmf
    result = flines(armf) - 1  # -1 from header
    if result != count:
        count_correct = False
        print(10 * "X")
        print(
            f"Warning: {'l_arm' if larm_flag else 'r_arm'} count is wrong, it is {result} instead of {count}"
        )
    else:
        print(f"{'l_arm' if larm_flag else 'r_arm'} count is OK {result} = {count}")
print(25 * "-")

if args.verify and not count_correct:
    exit(1)
