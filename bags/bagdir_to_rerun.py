#!/usr/bin/env python3
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import rerun as rr
from scipy.spatial.transform import Rotation

import reachy2_modelling.old as rm
import reachy2_modelling.old.rerun_utils as ru
from reachy2_modelling.old.symik import *
from reachy2_modelling.rerun_loader_urdf import URDFLogger

parser = argparse.ArgumentParser()
parser.add_argument("bagdir", type=str)
args = parser.parse_args()
print("bagdir:", args.bagdir)
print("urdf:", rm.urdf_path)

print("rr init...")
rr.init("Reachy Replay", spawn=True)
rr.send_blueprint(ru.blueprint())

print("urdflogger...")
urdf_logger = URDFLogger(rm.urdf_path)
print(urdf_logger.entity_to_transform)


scene = ru.Scene(args.bagdir)
print("scene.log()...")
scene.log(urdf_logger)
