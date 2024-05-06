#!/usr/bin/env python3
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import rerun as rr
from scipy.spatial.transform import Rotation

import reachy2_modelling.old.pin as oldrp
import reachy2_modelling.pin as rp
import reachy2_modelling.rerun_utils as ru
from reachy2_modelling.rerun_loader_urdf import URDFLogger

parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
parser.add_argument("--web", action="store_true", help="start rerun in web mode")
parser.add_argument("--old", action="store_true", help="use old urdf model")
args = parser.parse_args()
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
model_l_arm = rpp.model_l_arm
model_r_arm = rpp.model_r_arm
print("urdf:", urdf_path)

print("rr init...")

rr.init("Reachy Replay", spawn=not args.web)
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

scene = ru.Scene(args.bagdir, model_l_arm, model_r_arm)
print("scene.log()...")
scene.log(urdf_logger)
