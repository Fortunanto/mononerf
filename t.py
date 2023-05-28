from utils.utils_3d import *
from einops import *

intrinsics = torch.load("data/ball/data/intrinsics.pt")
poses = torch.load("data/ball/data/pose.pt")
pose = poses[0]
# assert False, f"poses.shape: {poses.shape} intrinsics.shape: {intrinsics.shape}" 
intrinsic = intrinsics[0]
intrinsic=  repeat(intrinsic, "a b -> n a b ", n=151)
pose =  repeat(pose, "a b -> n a b ", n=151)
print(intrinsic.shape, pose.shape)


# assert False, f"intrinsic {intrinsic}"
points = torch.randn(151,3,3).to(device="cuda")
points_2d = project_3d_to_2d_batch(points, intrinsics, poses)
assert False , f"points_2d.shape: {points_2d.shape}"

# assert False, f"pose.shape: {pose.shape} intrinsic {intrinsic.shape}"
# assert False, f"pose.shape: {pose.shape} intrinsics {intrinsics.shape}"
# print(project_3d_to_2d(np.array([-1,1,0]), intrinsic, pose))
# print(world_mat)\