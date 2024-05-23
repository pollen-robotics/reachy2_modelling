from ..kdl import KDLWrapper
from ..kdl_parser_py import urdf
from ..urdf import content_old as urdf_content

_, urdf_tree = urdf.treeFromString(urdf_content)

r_arm_world = KDLWrapper(urdf_tree, "world", "r_arm_tip")
r_arm_torso = KDLWrapper(urdf_tree, "torso", "r_arm_tip")

l_arm_world = KDLWrapper(urdf_tree, "world", "l_arm_tip")
l_arm_torso = KDLWrapper(urdf_tree, "torso", "l_arm_tip")

head_world = KDLWrapper(urdf_tree, "world", "head_tip")
head_torso = KDLWrapper(urdf_tree, "torso", "head_tip")
