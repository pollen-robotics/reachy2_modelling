#!/usr/bin/env python3
import argparse
import os
import subprocess

import yaml


def flines(fname):
    return sum(1 for _ in open(fname))


def blank(fname):
    with open(fname, "w") as f:
        f.write("")


def verify(source, lcount, rcount, lexpected, rexpected):
    lcount_correct = True
    rcount_correct = True

    def err_msg(arm, count, expected):
        print(f"Warning: {arm} count is wrong, it is {count} instead of {expected}")
        print(f"source of expected: {source}")
        print(10 * "X")
        return False

    def ok_msg(arm, count, expected):
        print(f"{arm} count is OK {count} = {expected} (source: {source})")
        return True

    func = err_msg
    if lcount == lexpected:
        func = ok_msg
    lcount_correct = func("l_arm", lcount, lexpected)

    func = err_msg
    if rcount == rexpected:
        func = ok_msg
    rcount_correct = func("r_arm", rcount, rexpected)

    return rcount_correct and lcount_correct


def metadata_arm_message_count(bagdir):
    metadatafile = os.path.join(bagdir, "metadata.yaml")
    with open(metadatafile, "r") as f:
        metadata = yaml.safe_load(f)

    larm_count = None
    rarm_count = None
    topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    for topic in topics:
        topicmeta = topic["topic_metadata"]
        if "l_arm" in topicmeta["name"]:
            larm_count = topic["message_count"]
        if "r_arm" in topicmeta["name"]:
            rarm_count = topic["message_count"]

    if larm_count is None or rarm_count is None:
        print(
            f"{metadatafile} does not contain message count for larm ({larm_count}) or rarm ({rarm_count})"
        )
        exit(1)

    return larm_count, rarm_count


def sqlite_arm_message_count(bagdir):
    sqlcmd = "SELECT topics.id, topics.name, count(messages.topic_id) from topics, messages where messages.topic_id=topics.id and (topics.name='/l_arm/target_pose' or topics.name='/r_arm/target_pose')  group by topics.name;"
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

    cmd = f'sqlite3 {args.bagdir}/{dbs[0]} "{sqlcmd}"'
    # print(cmd)
    rawstr = subprocess.check_output(cmd, shell=True).strip().decode("utf-8")
    larm_inline = ["l_arm" in line for line in rawstr.split("\n")]
    linecount_pertopic = [int(line.split("|")[-1]) for line in rawstr.split("\n")]
    assert len(larm_inline) == len(linecount_pertopic)
    assert len(larm_inline) == 2
    larm_count = linecount_pertopic[larm_inline.index(True)]
    rarm_count = linecount_pertopic[larm_inline.index(False)]

    return larm_count, rarm_count


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

count_correct = True
larm_csv_count = flines(larmf) - 1
rarm_csv_count = flines(rarmf) - 1
try:
    subprocess.check_output(["which", "sqlite3"])
except subprocess.CalledProcessError:
    print("sqlite3 not found, can't verify rosbag size")
else:
    larm_sql_msgcount, rarm_sql_msgcount = sqlite_arm_message_count(args.bagdir)
    count_correct = verify(
        source="sqlite",
        lcount=larm_csv_count,
        lexpected=larm_sql_msgcount,
        rcount=rarm_csv_count,
        rexpected=rarm_sql_msgcount,
    )
    print(25 * "-")


larm_meta_msgcount, rarm_meta_msgcount = metadata_arm_message_count(args.bagdir)
count_correct = verify(
    source="metadata.yaml",
    lcount=larm_csv_count,
    lexpected=larm_meta_msgcount,
    rcount=rarm_csv_count,
    rexpected=rarm_meta_msgcount,
)
print(25 * "-")

if args.verify and not count_correct:
    exit(1)
