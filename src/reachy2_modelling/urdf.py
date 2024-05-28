import tempfile
from copy import copy
from pathlib import Path

import numpy as np

original_urdf_path = str(
    Path(__file__).parent / "reachy2_rerun_test" / "reachy_v2_fix.urdf"
)
original_urdf_str = open(original_urdf_path).read()


def file_contents(filepath):
    with open(filepath, "r") as f:
        return f.read()


def fixed_mesh_paths(filepath):
    urdf_str = file_contents(filepath)

    toreplace = "reachy_description"
    urdf_str = urdf_str.replace(toreplace, str(Path(filepath).parent / toreplace))
    toreplace = "arm_description"
    urdf_str = urdf_str.replace(toreplace, str(Path(filepath).parent / toreplace))

    tmp = tempfile.NamedTemporaryFile(delete=False)
    urdf_path = tmp.name
    with open(urdf_path, "w") as f:
        f.write(urdf_str)
    return urdf_path, urdf_str


path_old = str(Path(__file__).parent / "old" / "reachy.urdf")
content_old = file_contents(path_old)


path_v2, content_v2 = fixed_mesh_paths(
    str(Path(__file__).parent / "reachy2_rerun_test" / "reachy_v2_fix.urdf")
)
# content_v2 = file_contents(path_v2)


path, content = fixed_mesh_paths(
    str(Path(__file__).parent / "reachy2_rerun_test" / "reachy_v3_fix.urdf")
)
# content = file_contents(path)

print(path)


def from_shoulder_offset(urdf_str, rpy=[10, 0, 15], distance=0):
    new_urdf_str = copy(urdf_str)
    roll = np.deg2rad(rpy[0])
    pitch = np.deg2rad(rpy[1])
    yaw = np.deg2rad(rpy[2])

    def rpystr(right_arm=True):
        reflect = 1
        if not right_arm:
            reflect = -1
        yoffset = -0.2 * reflect
        return f'<origin rpy="{reflect*(np.pi/2 + roll)} {pitch} {reflect*yaw}" xyz="0.0 {yoffset} 0.0"/>'

    #############################################################################
    # shoulder angle offset
    # rarm
    tosearch = (
        '<origin rpy="1.7453292519943295 0 0.2617993877991494" xyz="0.0 -0.2 0.0"/>'
    )
    assert new_urdf_str.count(tosearch) == 1
    toreplace = rpystr(right_arm=True)
    new_urdf_str = new_urdf_str.replace(tosearch, toreplace)

    # larm
    tosearch = (
        '<origin rpy="-1.7453292519943295 0 -0.2617993877991494" xyz="0.0 0.2 0.0"/>'
    )
    assert new_urdf_str.count(tosearch) == 1
    toreplace = rpystr(right_arm=False)
    new_urdf_str = new_urdf_str.replace(tosearch, toreplace)
    #############################################################################
    # shoulder distance offset
    for armstr in ["r", "l"]:
        tosearch = (
            f'<joint name="{armstr}_shoulder_dummy_out" type="fixed">\n    '
            '<origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/>'
        )
        assert new_urdf_str.count(tosearch) == 1
        yoffset = -distance if armstr == "r" else distance
        new_urdf_str = new_urdf_str.replace(
            tosearch, tosearch.replace("0 0 0", f"0 {yoffset} 0")
        )

    #############################################################################

    tmp = tempfile.NamedTemporaryFile(delete=False)
    new_urdf_file = tmp.name
    with open(new_urdf_file, "w") as f:
        f.write(new_urdf_str)
    print("new:", new_urdf_file)
    return new_urdf_str, new_urdf_file
