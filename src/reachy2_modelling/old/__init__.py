# TODO proper way to do it?
from pathlib import Path

# import reachy2_modelling.old.kdl
# import reachy2_modelling.old.pin
# import reachy2_modelling.old.rtb

urdf_path = str(Path(__file__).parent / "reachy.urdf")
urdf_content = ""
with open(urdf_path, "r") as f:
    urdf_content = f.read()
