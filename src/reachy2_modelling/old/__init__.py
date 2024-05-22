# TODO proper way to do it?
from pathlib import Path

import reachy2_modelling as r2
import reachy2_modelling.old.kdl
import reachy2_modelling.old.pin
import reachy2_modelling.old.rtb

urdf_path = str(Path(__file__).parent / "reachy.urdf")


def urdf_file_contents():
    with open(urdf_path, "r") as f:
        return f.read()


urdf_content = r2.LazyVar(lambda: urdf_file_contents)
