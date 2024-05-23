import tempfile
from pathlib import Path

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
