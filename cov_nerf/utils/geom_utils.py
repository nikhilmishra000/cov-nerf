import torch
import torch.nn.functional as F

from torch_utils import expand_dim


def rotate_vectors(pose: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    rotmat = pose[..., :3, :3]
    return torch.einsum("...ij,...j->...i", rotmat, vecs)


def transform_points(pose: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    rotmat = pose[..., :3, :3]
    pos = pose[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rotmat, pts) + pos


def project_points_to_pixels(
    pts: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    px = pts.matmul(intrinsics.transpose(-2, -1))
    px = px[..., 0:2] / px[..., 2, None]
    return px


@torch.jit.script
def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (does not need to be normalized beforehand) to rotation matrix."""
    r, i, j, k = torch.unbind(quat, -1)
    two_s = 2.0 / (quat * quat).sum(-1)
    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return o.reshape(quat.shape[:-1] + (3, 3))


def to_pose(pos=None, quat=None) -> torch.Tensor:
    mat = torch.eye(4)
    if pos is not None:
        mat[:3, 3] = torch.tensor(pos)

    if quat is not None:
        quat = torch.tensor(quat)
        rotmat = quat_to_rotation_matrix(quat)
        mat[:3, :3] = rotmat

    return mat


def depth2cloud(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Convert depth image to XYZ point cloud.

    Args
    ----
        depth : Tensor[DeviceT, FloatT, SHAPE_HW]
            Contains depth value for every pixel in the image.
        intrinsic : Tensor[DeviceT, FloatT, SHAPE_33]
            Master camera intrinsic.

    Returns
    -------
        pts : Tensor[DeviceT, FloatT, SHAPE_HW3]
            Point cloud.
    """
    device = depth.device
    height, width = int(depth.size(-2)), int(depth.size(-1))
    vu = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        ),
        dim=0,
    )
    uv1 = F.pad(vu.flip(0), (0, 0, 0, 0, 0, 1), value=1)
    pts = torch.einsum("ij,jhw->hwi", intrinsic.inverse(), uv1 * depth.unsqueeze(0))
    return pts
