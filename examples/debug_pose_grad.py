from curobo.types.math import Pose
from curobo.geom.transform import pose_inverse
import torch

position = torch.tensor([0.0, 0.0, 0.0], device="cuda")
quaternion = torch.tensor([1, 0., 0.5, 1.], device="cuda")
pose = Pose(
    position=position,
    quaternion=quaternion,
    normalize_rotation=True
)
position = pose.position.requires_grad_(True)
quaternion = pose.quaternion.requires_grad_(True)
print(f'position: {position}')
print(f'quaternion: {quaternion}')

def loss_f(position, quaternion, idx=0):
    inv_pos, inv_quat = pose_inverse(position, quaternion)
    return inv_quat[0, idx]  + inv_pos.mean()

position.grad = None
quaternion.grad = None
loss = loss_f(position, quaternion)
loss.backward()

print(f'position.grad: {position.grad}')
print(f'quaternion.grad: {quaternion.grad}')

success = torch.autograd.gradcheck(loss_f, inputs=(position, quaternion, 0), eps=1e-3)
