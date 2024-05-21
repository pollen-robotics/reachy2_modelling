#!/usr/bin/env python

from pathlib import Path

import pinocchio as pin

from ..pin import Models

urdf_path = str(Path(__file__).parent / "reachy.urdf")
robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
urdf_str = open(urdf_path).read()

models = Models(urdf_str)
