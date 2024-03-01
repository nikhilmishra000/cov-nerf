from typing import Any

import attr
import torch
import torch.nn.functional as F

from cov_nerf.utils.geom_utils import project_points_to_pixels
from torch_utils import DeviceTransferableMixin

IndexOrSlice = int | slice | torch.Tensor


@attr.s(kw_only=True, frozen=True, repr=False)
class Frustum(DeviceTransferableMixin):
    """A frustum is a trapezoidal volume in 3D space, defined by a 2D bounding box, camera intrinsic, and near/far planes."""

    roi: torch.Tensor = attr.ib()
    """Bounding boxes in the image plane. Shape (..., 4), dtype float32."""

    near_plane: torch.Tensor = attr.ib()
    """Shape (...) matching `roi`, dtype float32."""

    far_plane: torch.Tensor = attr.ib()
    """Shape (...) matching `roi`, dtype float32."""

    intrinsic: torch.Tensor = attr.ib()
    """Shape (3, 3), dtype float32."""

    @property
    def shape(self) -> torch.Size:
        return self.roi.shape[:-1]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def device(self) -> torch.device:
        return self.roi.device

    def __getitem__(self, key: Any) -> "Frustum":
        return Frustum(
            roi=self.roi[key],
            near_plane=self.near_plane[key],
            far_plane=self.far_plane[key],
            intrinsic=self.intrinsic,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(shape={list(self.shape)}, device={self.device}) at {hex(id(self))}>"

    def get_corner_points(self) -> torch.Tensor:
        arange = torch.linspace(0, 1, 2, device=self.device)
        lams = torch.stack(
            torch.meshgrid(arange, arange, indexing="ij"), dim=-1
        ).flatten(0, -2)
        corners_px = torch.einsum(
            "...i,bi->...bi", self.roi[..., 0:2], 1 - lams
        ) + torch.einsum("...i,bi->...bi", self.roi[..., 2:4], lams)
        corners_px = F.pad(corners_px, (0, 1), value=1.0)
        corners_uv1 = torch.cat(
            [
                corners_px * self.near_plane[..., None, None],
                corners_px * self.far_plane[..., None, None],
            ],
            dim=-2,
        )
        corners_cam = torch.einsum(
            "ij,...j->...i", self.intrinsic.inverse(), corners_uv1
        )
        return corners_cam

    def to_grid(
        self, voxel_size: torch.Tensor, alignment: str = "center"
    ) -> torch.Tensor:
        """Compute a dense grid of 3D points inside this frustum.

        Parameters
        ----------
        voxel_size : torch.Tensor
            Shape (3), dtype int64.

        Returns
        -------
        torch.Tensor
            Shape (*self.shape, *voxel_size, 3), dtype float32.
            The entries correspond to the xyz coordinates of the (trapezoidal) voxel centers, in the camera frame.
            The depth values are spaced linearly (this may change / be configurable in the future).
        """
        if alignment == "center":
            half_bin = 0.5 / voxel_size
            lb, ub = half_bin, 1 - half_bin
        elif alignment == "endpoint":
            lb = torch.zeros_like(voxel_size, dtype=torch.float32)
            ub = torch.ones_like(voxel_size, dtype=torch.float32)
        else:
            raise NotImplementedError(alignment)

        device = self.device
        lams = torch.stack(
            torch.meshgrid(
                torch.linspace(lb[0], ub[0], voxel_size[0], device=device),
                torch.linspace(lb[1], ub[1], voxel_size[1], device=device),
                torch.linspace(lb[2], ub[2], voxel_size[2], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        lb = torch.cat(
            [self.near_plane.unsqueeze(-1), self.roi[..., 0:2].flip(-1)], dim=-1
        )
        ub = torch.cat(
            [self.far_plane.unsqueeze(-1), self.roi[..., 2:4].flip(-1)], dim=-1
        )
        zvu = torch.einsum("...i,dhwi->...dhwi", lb, 1 - lams) + torch.einsum(
            "...i,dhwi->...dhwi", ub, lams
        )
        uv1 = F.pad(zvu[..., 1:3].flip(-1), (0, 1), value=1.0)
        z = zvu[..., 0, None]
        pts_cam = torch.einsum("ij,...j->...i", self.intrinsic.inverse(), uv1 * z)
        return pts_cam

    def normalize_coords(self, pts_cam: torch.Tensor) -> torch.Tensor:
        px = project_points_to_pixels(pts_cam, self.intrinsic)
        lb, ub = self.roi[..., 0:2], self.roi[..., 2:4]
        uv = (px - lb) / (ub - lb)
        z = (pts_cam[..., 2] - self.near_plane) / (self.far_plane - self.near_plane)
        uvz = torch.cat([uv, z.unsqueeze(-1)], dim=-1)
        return uvz
