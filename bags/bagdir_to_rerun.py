#!/usr/bin/env python3
import argparse

import rerun as rr

import reachy2_modelling as r2
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
torso_entity = "world/world_joint/base_link/back_bar_joint/back_bar/torso_base/torso"
urdf_path = r2.urdf.path
models = r2.pin.models
if args.old:
    urdf_path = r2.urdf.path_old
    torso_entity = "world/world_joint/torso"
    models = r2.old.pin.models
########################

shoulder_offset = None
if args.offset is not None:
    if args.old:
        print("Error: --old cannot be used with --shoulder-offset")
        exit(1)
    rpy = args.offset.split(",")
    if len(rpy) != 3:
        print(
            "Error: --offset command must provide 3 elements like (in degrees): --shoulder-offset=10,0,15"
        )
        exit(1)
    rpy = [float(x) for x in rpy]
    shoulder_offset = rpy
    roll, pitch, yaw = [*rpy]
    print(roll, pitch, yaw)
    models, urdf_str, urdf_path = r2.pin.PinModels.from_shoulder_offset(
        roll, pitch, yaw
    )
    print("shoulder offset:", shoulder_offset)

model_l_arm = models.l_arm
model_r_arm = models.r_arm

print("urdf:", urdf_path)

print("rr init...")

if args.save:
    recext = ".rrd"
    if not args.save.endswith(recext):
        args.save += recext
    args.save = f"{args.bagdir}_{args.save}"
    print(f"saving recording to {args.save}...")
    rr.save(args.save, default_blueprint=r2.rerun_utils.blueprint())

rr.init("reachy_replay2", spawn=not args.web)
if args.web:
    rr.serve()
rr.send_blueprint(r2.rerun_utils.blueprint())

print("urdflogger...")
urdf_logger = URDFLogger(urdf_path, torso_entity)

scene = r2.rerun_utils.Scene(
    args.bagdir, model_l_arm, model_r_arm, shoulder_offset=shoulder_offset
)
print("scene.log()...")
scene.log(urdf_logger)
