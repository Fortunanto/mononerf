import numpy as np
from scipy.spatial.transform import Rotation as R

def decompose_matrix(E):
    """Decompose the extrinsic matrix into rotation matrix and translation vector."""
    R = E[:3, :3]
    t = E[:3, 3]
    return R, t

def compose_matrix(R, t):
    """Compose the extrinsic matrix from rotation matrix and translation vector."""
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    return E

def slerp(q1, q2, t):
    """Spherical linear interpolation of quaternions."""
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (1 - t) * q1 + t * q2
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sqrt(1.0 - dot*dot)
    theta = theta_0 * t
    w = np.sin(theta) / sin_theta_0
    return (np.sin(theta_0 - theta) / sin_theta_0) * q1 + w * q2

def interpolate_matrices(E1, E2, s):
    """Interpolate between two extrinsic matrices."""

    # Decompose the matrices
    R1, t1 = decompose_matrix(E1)
    R2, t2 = decompose_matrix(E2)

    # Interpolate translation
    t = (1 - s) * t1 + s * t2

    # Convert rotation matrices to quaternions and interpolate
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()
    q = slerp(q1, q2, s)

    # Convert the interpolated quaternion back to a rotation matrix
    R_mat = R.from_quat(q).as_matrix()

    # Compose the interpolated matrix
    E = compose_matrix(R_mat, t)

    return E

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def normalize(x):
    return x / (np.linalg.norm(x))

def calculate_sc(pose_bound,bd_factor):
    return 1. if bd_factor is None else 1./(np.percentile(pose_bound[:, -2], 5) * bd_factor)

def poses_avg(poses):

    hwf = poses[ :3, -1:].squeeze()

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w
def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses
poses_bounds = np.load("/home/yiftach/main/Research/MonoNeRF/data/Balloon1/poses_bounds.npy")
poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
sc = calculate_sc(poses_bounds,0.9)
print(poses)

poses[...,:3,3] *= sc
hwfs = poses[...,-1]*1./2

poses = np.concatenate([poses[..., 1:2],
                    -poses[..., 0:1],
                    poses[...,2:]], -1)
poses = recenter_poses(poses)[...,:-1]
# poses = np.stack([recenter_poses(pose) for pose in poses])[...,:-1]
last_row = np.array([0,0,0,1])
poses = np.concatenate([poses, np.tile(last_row, [poses.shape[0],1,1])], axis=1)
pose1 = poses[0]
pose2 = poses[1]
p_interp = interpolate_matrices(pose1,pose2,1/2)
np.save("interpolated_balloon1_pose0pose1.npy",p_interp)
# print(p_interp)