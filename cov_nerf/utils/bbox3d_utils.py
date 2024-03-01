from typing import Sequence, Tuple

import torch

from cov_nerf.utils.geom_utils import quat_to_rotation_matrix, transform_points


def get_uniform_rotations(n: int = 10000, device: torch.device = "cpu") -> torch.Tensor:
    """Compute random rotations by uniformly sampling quaternions.

    Source: https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf

    Parameters
    ----------
    n: int (default = 10000)
        How many rotations to generate.
    device: torch.device (default = "cpu")

    Returns
    -------
    rots: Tensor[DeviceT, Float32, SHAPE_B33]
        Random rotations.
    """
    # TODO: Investigate why using torch here leads to `RNGState.set_rng(get_bbox3d_from_occupancy)` behaving non-deterministically.
    s = torch.rand([n], device=device)
    sigma_1 = torch.sqrt(1 - s)
    sigma_2 = torch.sqrt(s)
    theta_1 = 2 * torch.pi * torch.rand([n], device=device)
    theta_2 = 2 * torch.pi * torch.rand([n], device=device)
    w = torch.cos(theta_2) * sigma_2
    x = torch.sin(theta_1) * sigma_1
    y = torch.cos(theta_1) * sigma_1
    z = torch.sin(theta_2) * sigma_2
    quats = torch.stack([w, x, y, z], axis=1)
    all_rots = quat_to_rotation_matrix(quats)
    return all_rots


def get_bbox3d_from_occupancy(
    points_list: Sequence[torch.Tensor],
    max_points: int = 2000,
    num_rotations: int = 10000,
    eps_top: float = 0.0,
    eps_bot: float = 0.0,
    bbox3d_sampling_batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = points_list[0].device
    rots = get_uniform_rotations(device=device, n=num_rotations)

    all_points = []
    assert (1 - eps_top) > eps_bot

    valids = torch.ones(len(points_list), device=device, dtype=torch.bool)
    # We don't want to use all the points that occupancy gives us for speed reasons.
    # The for-loop below does some randomized filtering of the points, and optionally applies
    # some quantile-based filtering of the point cloud.
    for idx, points in enumerate(points_list):
        if points.size(0) == 0:
            # Append some fake values and set the bounding box to be invalid
            valids[idx] = False
            all_points.append(
                torch.ones(
                    size=(max_points, 3), device=points.device, dtype=points.dtype
                )
            )
        else:
            valids[idx] = True
            if eps_bot > 0 or eps_top > 0:
                top_kth = (
                    points[..., -1:]
                    .kthvalue(int(points.shape[-2] * (1 - eps_top)), dim=0)
                    .values
                )
                bot_kth = (
                    points[..., -1:]
                    .kthvalue(int(points.shape[-2] * eps_bot) + 1, dim=0)
                    .values
                )
                points = points[(points[:, -1] >= bot_kth) & (points[:, -1] <= top_kth)]

            if len(points) > max_points:
                perm = torch.randperm(points.size(0), device=device)
                idx = perm[:max_points]
                all_points.append(points[idx])
            else:
                idx = torch.randint(
                    low=0, high=points.size(0), size=(max_points,), device=device
                )
                all_points.append(points[idx])

    # For each random rotation that we sampled in the very first step above, we compute the min/max
    # for each dimension over the subsampled occupancy point cloud. In some sense, we are fitting a tight bounding box.
    # This provides us both with the dimensions & center of the bounding box.
    all_points = torch.stack(all_points)
    centered_points = all_points - all_points.mean(1).unsqueeze(1)
    rots_a_b = rots
    rots_b_a = rots_a_b.transpose(1, 2)
    num_objs = centered_points.size(0)
    batch_maxs, batch_mins = [], []
    inds = torch.arange(num_objs, dtype=torch.int64, device=centered_points.device)
    # We use the for-loop because we don't want to risk OOM errors.
    for ind in inds.split(bbox3d_sampling_batch_size):
        curr_transformed_points = torch.einsum(
            "bij, knj->kbni", rots_b_a, centered_points[ind]
        )
        batch_maxs.append(curr_transformed_points.max(-2).values)
        batch_mins.append(curr_transformed_points.min(-2).values)

    # Compute the min/max values for each rotation
    max_vals, min_vals = torch.cat(batch_maxs, dim=0), torch.cat(batch_mins, dim=0)

    # Get the bbox dimensions
    extents = (max_vals - min_vals).abs()
    # Find the minium volume bbox
    volumes = extents.prod(-1)
    min_vol_ind = volumes.min(1).indices

    # Get the center of bbox in the local frame.
    center = (
        max_vals[torch.arange(len(min_vol_ind), device=device), min_vol_ind]
        + min_vals[torch.arange(len(min_vol_ind), device=device), min_vol_ind]
    ).div(2)
    # Get the center of bbox in the camera frame.
    new_centers = torch.einsum(
        "kji,kj->ki", rots_b_a[min_vol_ind], center
    ) + all_points.mean(
        1
    )  # inverse -> ji instead of ij
    new_rots = rots_a_b[min_vol_ind]
    new_extents = extents[torch.arange(len(min_vol_ind), device=device), min_vol_ind]

    new_poses = torch.eye(4, device=device)[None].repeat(num_objs, 1, 1)
    new_poses[:, :3, :3] = new_rots
    new_poses[:, :3, 3] = new_centers
    return new_poses, new_extents


def box_contains_points(
    pose: torch.Tensor, dimension: torch.Tensor, points: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Check if points are contained inside a bounding box.

    The way we perform this test:
      1. Transform `points` into the bbox frame (defined by `bbox.pose`).
      2. Normalize the transformed points by `0.5 * bbox.dimension`.
      3. If the normalized points are within the cube with vertices `{-1, 1}^3`, then the points are within the bbox.

    Parameters
    ----------
    bbox : BBox3D
    points : torch.Tensor, shape(..., 3), floating dtype.
    eps : float
        Relax the cube from `{-1, 1}^3` to `{-(1+eps), 1+eps}^3`.

    Returns
    -------
    torch.Tensor
        The same leading shape as `points`, dtype bool.
    """
    box_from_world = pose.inverse()
    points_in_box = transform_points(box_from_world, points) / (0.5 * dimension)
    return (points_in_box.abs() < (1.0 + eps)).all(-1)
