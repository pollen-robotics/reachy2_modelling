#!/usr/bin/env python3
import argparse

import rerun as rr

import reachy2_modelling.old.pin as oldrp
import reachy2_modelling.pin as rp
import reachy2_modelling.rerun_utils as ru
from reachy2_modelling.rerun_loader_urdf import URDFLogger

parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
parser.add_argument("--web", action="store_true", help="start rerun in web mode")

group = parser.add_mutually_exclusive_group()
group.add_argument("--old", action="store_true", help="use old urdf model")
group.add_argument(
    "--offset",
    type=str,
    help="should offset of like roll,pitch,yaw (in degrees, BEWARE spaces)",
)
parser.add_argument(
    "--save", type=str, help="save the recording to a file: --save=filename"
)
args = parser.parse_args()
# print(args)
# exit(0)

print("bagdir:", args.bagdir)

########################
# use new or old model
rpp = rp
torso_entity = "world/world_joint/base_link/back_bar_joint/back_bar/torso_base/torso"
if args.old:
    rpp = oldrp
    torso_entity = "world/world_joint/torso"
########################

urdf_path = rpp.urdf_path
models = rpp.models
shoulder_offset = None
if args.offset is not None:
    rpy = args.offset.split(",")
    if len(rpy) != 3:
        print(
            "Error: --offset command must provide 3 elements like (in degrees): --offset=10,0,15"
        )
        exit(1)
    rpy = [float(x) for x in rpy]
    shoulder_offset = rpy
    roll, pitch, yaw = [*rpy]
    print(roll, pitch, yaw)
    urdf_str, urdf_path = rpp.offset_urdf(roll, pitch, yaw)
    models = rpp.Models(urdf_str)

model_l_arm = models.l_arm
model_r_arm = models.r_arm

print("shoulder offset:", shoulder_offset)
print("urdf:", urdf_path)

print("rr init...")

if args.save:
    recext = ".rrd"
    if not args.save.endswith(recext):
        args.save += recext
    args.save = f"{args.bagdir}_{args.save}"
    print(f"saving recording to {args.save}...")
    rr.save(args.save, default_blueprint=ru.blueprint())

rr.init("reachy_replay2", spawn=not args.web)
if args.web:
    rr.serve()
rr.send_blueprint(ru.blueprint())

print("urdflogger...")
urdf_logger = URDFLogger(urdf_path, torso_entity)
# urdfp_logger = URDFLogger("/home/user/pol/python-example-droid-dataset/franka_description/panda.urdf")
# print(urdf_logger.entity_to_transform)
# urdf_logger.log()
# urdfp_logger.log()
# urdf_logger.log_joint()
# urdf = urdf_logger.urdf
# urdfp = urdfp_logger.urdf
# rjoints = [x for x in urdf_logger.joint_entity_paths.keys() if 'r_' in x]

scene = ru.Scene(args.bagdir, model_l_arm, model_r_arm, shoulder_offset=shoulder_offset)
print("scene.log()...")
scene.log(urdf_logger)
