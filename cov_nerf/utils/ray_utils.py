from typing import Tuple

import attr
import torch
import torch.nn.functional as F

from cov_nerf.utils.bbox3d_utils import box_contains_points
from cov_nerf.utils.geom_utils import rotate_vectors, transform_points
from torch_utils import DeviceTransferableMixin


@attr.s(kw_only=True, frozen=True, repr=False)
class Rays(DeviceTransferableMixin):
    o: torch.Tensor = attr.ib()
    d: torch.Tensor = attr.ib()

    @property
    def shape(self):
        return self.o.shape[:-1]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def device(self):
        return self.o.device

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(shape={list(self.shape)}, device={self.device}) at {hex(id(self))}>"

    def __getitem__(self, sli):
        if not isinstance(sli, tuple):
            sli = (sli,)

        sli = (*sli, slice(None))
        return attr.evolve(self, o=self.o[sli], d=self.d[sli])

    def reshape(self, shape: torch.Size) -> "Rays":
        return attr.evolve(
            self, o=self.o.reshape(*shape, 3), d=self.d.reshape(*shape, 3)
        )

    def get_points(self, t: torch.Tensor) -> torch.Tensor:
        return self.o + torch.einsum("...,...i->...i", t, self.d)

    def transform(self, transform: torch.Tensor) -> "Rays":
        return Rays(
            o=transform_points(transform, self.o), d=rotate_vectors(transform, self.d)
        )

    @classmethod
    def from_camera(
        cls,
        intrinsic: torch.Tensor,
        world_from_cam: torch.Tensor,
        im_size: Tuple[int, int],
    ) -> "Rays":
        device = intrinsic.device
        v, u = torch.meshgrid(
            torch.arange(im_size[0], device=device, dtype=torch.float32),
            torch.arange(im_size[1], device=device, dtype=torch.float32),
            indexing="ij",
        )
        px = torch.stack([u, v], dim=-1)
        return cls.from_pixels(intrinsic, world_from_cam, px)

    @classmethod
    def from_pixels(
        cls,
        intrinsic: torch.Tensor,
        world_from_cam: torch.Tensor,
        px: torch.Tensor,
    ) -> "Rays":
        uv1 = F.pad(px, (0, 1), value=1.0)
        rays_o_cam = torch.einsum(
            "ij,...j->...i", intrinsic.inverse(), torch.zeros_like(uv1)
        )
        rays_op1_cam = torch.einsum("ij,...j->...i", intrinsic.inverse(), uv1)
        rays_o = transform_points(world_from_cam, rays_o_cam)
        rays_op1 = transform_points(world_from_cam, rays_op1_cam)
        rays_d = F.normalize(rays_op1 - rays_o, p=2, dim=-1)
        return cls(o=rays_o, d=rays_d)


@attr.s(kw_only=True, frozen=True, repr=False)
class RayBounds(DeviceTransferableMixin):
    t_near: torch.Tensor = attr.ib()
    t_far: torch.Tensor = attr.ib()

    @t_near.validator
    def validate_t_near(self, attribute, value):
        assert self.t_near.shape == self.t_far.shape
        assert self.t_near.device == self.t_far.device

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return self.t_near.shape

    @property
    def device(self):
        return self.t_near.device

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(shape={list(self.shape)}, device={self.device}) at {hex(id(self))}>"

    def __getitem__(self, sli):
        return attr.evolve(self, t_near=self.t_near[sli], t_far=self.t_far[sli])

    def sample(self, n_samples: int, strategy: str = "stratified"):
        if strategy == "stratified":
            lam = torch.linspace(
                1 / n_samples, 1 - 1 / n_samples, n_samples, device=self.device
            )
            lam = lam + torch.rand(*self.shape, n_samples, device=self.device).sub(
                0.5
            ).div(n_samples)
        else:
            raise NotImplementedError(strategy)

        return torch.einsum("...,...i->...i", self.t_near, 1 - lam) + torch.einsum(
            "...,...i->...i", self.t_far, lam
        )

    def isfinite(self):
        return self.t_near.isfinite() & self.t_far.isfinite()


def ray_box_intersection(
    rays: Rays, box_pose: torch.Tensor, box_dimension: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Compute intersections between rays and a bounding box."""
    obj_from_world = box_pose.inverse()
    rays_in_obj = rays.transform(obj_from_world)

    w_batch = torch.eye(3, device=rays.device).repeat(2, 1)
    halfsize = box_dimension / 2
    c_batch = torch.cat([halfsize, -halfsize])
    top = c_batch - torch.einsum("bi,...i->...b", w_batch, rays_in_obj.o)
    bottom = torch.einsum("bi,...i->...b", w_batch, rays_in_obj.d)
    t_hit = top / bottom

    t_hit_sorted, _ = t_hit.sort(dim=-1)
    p_hit_sorted = rays[..., None].get_points(t_hit_sorted)
    within_box = box_contains_points(box_pose, box_dimension, p_hit_sorted, eps=eps)
    valid = within_box & (t_hit_sorted >= 0)

    # this should be `first_along_dim()`, but `torch.take_along_dim()` doesn't allow -1.
    first_hit_idx = valid.byte().argmax(-1)
    first_hit = t_hit_sorted.take_along_dim(first_hit_idx[..., None], dim=-1).squeeze(
        -1
    )
    first_hit = first_hit.masked_fill(~valid.any(-1), torch.nan)
    return first_hit.view(rays.shape)
