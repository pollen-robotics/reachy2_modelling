from ..pin import PinModels, PinWrapperArm, PinWrapperHead
from ..urdf import content_old as urdf_content

models = PinModels(urdf_content)

l_arm = PinWrapperArm("l_arm", custom_model=models.l_arm)
r_arm = PinWrapperArm("r_arm", custom_model=models.r_arm)
head = PinWrapperHead(custom_model=models.head)
