from collections import defaultdict
from typing import Dict, Sequence, Tuple

import attr
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from cov_nerf.backbone import Backbone, PositionalEncoding1d
from cov_nerf.nerformer_blocks import (
    CollapseCamsBlock,
    NerformerBlock,
    RayTransformerBlock,
)
from cov_nerf.utils.bbox3d_utils import get_bbox3d_from_occupancy
from cov_nerf.utils.frustum import Frustum
from cov_nerf.utils.geom_utils import (
    project_points_to_pixels,
    rotate_vectors,
    transform_points,
)
from cov_nerf.utils.ray_utils import RayBounds, Rays, ray_box_intersection
from layers.conv_blobs import ConvBlobSpec2d, ConvBlobSpec3d
from layers.conv_layer import ConvArgs3d, ConvLayer, ConvLayerSpec, LayerOrder
from layers.mlp import MLP
from layers.nonlinearity import LeakyReluArgs
from layers.normalization import GroupNormArgs
from layers.residual_block import ResidualBlock, ResidualMode
from layers.unet import UNetSimple
from torch_utils import (
    BatchApply,
    DeviceTransferableMixin,
    Lambda,
    batch_interp2d,
    batch_interp3d,
    expand_dim,
    interp3d,
)


@attr.s(kw_only=True, frozen=True)
class NerfObject(DeviceTransferableMixin):
    pose: torch.Tensor = attr.ib()
    dimension: torch.Tensor = attr.ib()
    voxel_probs: torch.Tensor = attr.ib()
    feature_vol: torch.Tensor = attr.ib()
    feature_vol_highres: torch.Tensor = attr.ib()

    def ray_bounds(self, rays: Rays) -> RayBounds:
        t_near = ray_box_intersection(rays, self.pose, self.dimension, eps=1e-3)
        t_max = t_near + 10 * self.dimension.abs().sum(-1)
        t_far = t_max - ray_box_intersection(
            Rays(o=rays.get_points(t_max), d=-rays.d),
            self.pose,
            self.dimension,
            eps=1e-3,
        )
        return RayBounds(t_near=t_near, t_far=t_far)

    @property
    def cam_from_obj(self) -> torch.Tensor:
        return self.pose

    @property
    def obj_from_cam(self) -> torch.Tensor:
        return self.cam_from_obj.inverse()

    def normalize_coords(self, pts_cam: torch.Tensor) -> torch.Tensor:
        pts_obj = transform_points(self.obj_from_cam, pts_cam)
        pts_normed = 0.5 + (pts_obj / self.dimension)
        return pts_normed

    def voxel_density(self, pts_cam: torch.Tensor) -> torch.Tensor:
        uvz = self.normalize_coords(pts_cam)
        voxel_shape = torch.tensor(
            self.voxel_probs.shape, device=self.voxel_probs.device
        )
        interp_probs = interp3d(
            self.voxel_probs[None],
            uvz.flip(-1) * voxel_shape.sub(1),
        ).squeeze(-1)
        return interp_probs

    def interp_feats(self, pts_cam: torch.Tensor) -> torch.Tensor:
        uvz = self.normalize_coords(pts_cam)
        voxel_shape = torch.tensor(
            self.feature_vol.shape[-3:], device=self.feature_vol.device
        )
        vol_feats = interp3d(self.feature_vol, uvz.flip(-1) * voxel_shape.sub(1))
        return vol_feats

    def interp_feats_highres(self, pts_cam: torch.Tensor) -> torch.Tensor:
        uvz = self.normalize_coords(pts_cam)
        voxel_shape = voxel_shape = torch.tensor(
            self.feature_vol_highres.shape[-3:], device=self.feature_vol.device
        )
        vol_feats = batch_interp3d(
            self.feature_vol_highres,
            expand_dim(
                uvz.flip(-1) * voxel_shape.sub(1),
                dim=0,
                shape=[len(self.feature_vol_highres)],
            ),
        )
        return vol_feats.movedim(0, -2)

    def save(self, path: str) -> None:
        torch.save(attr.asdict(self, recurse=False), path)

    @classmethod
    def load(cls, path: str) -> "NerfObjectBbox":
        kvs = torch.load(path, map_location="cpu")
        return cls(**kvs)


class BackgroundRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        c = 256
        n_bases = 6
        self.backbone = Backbone()
        self.pos_enc = PositionalEncoding1d(n_bases)

        c_in = 3 + self.backbone.channels[0] + 4 * n_bases * 2
        self.core = nn.Sequential(
            nn.Linear(c_in, c),
            NerformerBlock(c, 8, 16, norm=True),
            Lambda(lambda x: x.mean(-2)),
            RayTransformerBlock(c, 8, 16, norm=True),
            MLP(c, [4], nonlin_args=LeakyReluArgs()),
        )

    def interp_feats(
        self,
        rgbs: torch.Tensor,
        feats: torch.Tensor,
        stride: int,
        intrinsics: torch.Tensor,
        cam_from_ref: torch.Tensor,
        rays: Rays,
        t: torch.Tensor,
    ) -> torch.Tensor:
        p_ref = rays[..., None].get_points(t)
        p_cam = transform_points(cam_from_ref[:, None, None], p_ref[None])
        px_cam = project_points_to_pixels(p_cam, intrinsics[:, None])
        interp_rgb = batch_interp2d(rgbs, px_cam)
        interp_feats = batch_interp2d(feats / stride, px_cam / stride)
        viewdir_cam = F.normalize(
            p_ref[None] - cam_from_ref[:, None, None, :3, 3], dim=-1
        )
        return torch.cat(
            [
                interp_rgb,
                interp_feats,
                self.pos_enc(viewdir_cam).flatten(-2, -1),
                expand_dim(self.pos_enc(t), dim=0, shape=[len(feats)]),
            ],
            dim=-1,
        ).movedim(0, -2)

    def forward(
        self,
        rgbs: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_from_ref: torch.Tensor,
        near_plane: float,
        far_plane: int,
        rays: Rays,
        n_ray_samples,
    ) -> Dict[str, torch.Tensor]:
        feats_lst = self.backbone(rgbs[1:])
        bounds = RayBounds(
            t_near=(near_plane - rays.o[..., 2]) / rays.d[..., 2],
            t_far=(far_plane - rays.o[..., 2]) / rays.d[..., 2],
        )

        t = bounds.sample(n_ray_samples)
        x = self.interp_feats(
            rgbs[1:],
            feats_lst[0],
            self.backbone.strides[0],
            intrinsics[1:],
            cam_from_ref[1:],
            rays,
            t,
        )
        xx = self.core(x)
        log_sigma, radiance = xx[..., 0], xx[..., 1:4].sigmoid()
        out = {
            "t": t,
            "alpha": log_sigma.softmax(-1),
            "radiance": radiance,
        }
        return out


@attr.s(kw_only=True, eq=False, repr=False)
class ObjectRenderer(nn.Module):
    input_channels: int = attr.ib()
    hidden_channels: int = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()
        c_in, c = self.input_channels, self.hidden_channels
        self.density_mlp = MLP(c_in + 1, [c, 1], nonlin_args=LeakyReluArgs())
        self.radiance = nn.Sequential(
            nn.Linear(c_in + 10, c),
            NerformerBlock(c, 8, 16),
            Lambda(lambda x: x.mean(-2)),
            RayTransformerBlock(c, 8, 16),
            MLP(c, [3], nonlin_args=LeakyReluArgs()),
        )

    def forward(
        self,
        rays: Rays,
        objects: Sequence[NerfObject],
        *,
        n_samples_primary: int,
        n_samples_importance: int,
        occ_thresh: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        n_objects = len(objects)
        samples_per_ray = n_samples_primary + n_samples_importance
        shape = torch.Size([n_objects]) + rays.shape + torch.Size([samples_per_ray])
        device = rays.device

        total_t = torch.zeros(shape, device=device)
        total_occ = torch.zeros(shape, device=device)
        total_alpha = torch.zeros(shape, device=device)
        total_radiance = torch.zeros(shape + torch.Size([3]), device=device)
        for obj_idx, obj in enumerate(objects):
            ray_bounds = obj.ray_bounds(rays)
            m = ray_bounds.isfinite()
            if not m.any():
                continue

            obj_rays, obj_bounds = rays[m], ray_bounds[m]
            t_coarse = obj_bounds.sample(n_samples_primary)
            p_coarse = obj_rays[..., None].get_points(t_coarse)
            x_coarse = obj.interp_feats(p_coarse)
            occ_coarse = obj.voxel_density(p_coarse)

            if n_samples_importance > 0:
                probs = occ_coarse[..., :-1] + occ_coarse[..., 1:]
                inds = (
                    D.Categorical(probs=probs.clamp(min=1e-12))
                    .sample([n_samples_importance])
                    .moveaxis(0, -1)
                )
                lam = torch.rand_like(inds, dtype=torch.float32)
                t_fine = lam * t_coarse.take_along_dim(inds, dim=-1) + (
                    1 - lam
                ) * t_coarse.take_along_dim(inds + 1, dim=-1)
                p_fine = obj_rays[..., None].get_points(t_fine)
                x_fine = obj.interp_feats(p_fine)
                occ_fine = obj.voxel_density(p_fine)

                t, permute = torch.cat([t_coarse, t_fine], dim=-1).sort()
                occ = torch.cat([occ_coarse, occ_fine], dim=-1).take_along_dim(
                    permute, dim=-1
                )
                p = torch.cat([p_coarse, p_fine], dim=-2).take_along_dim(
                    permute.unsqueeze(-1), dim=-2
                )
                x = torch.cat([x_coarse, x_fine], dim=-2).take_along_dim(
                    permute.unsqueeze(-1), dim=-2
                )

            else:
                t, occ, p, x = t_coarse, occ_coarse, p_coarse, x_coarse

            xx = torch.cat([x, occ.unsqueeze(-1)], dim=-1)
            sigma = self.density_mlp(xx).squeeze(-1).sigmoid()
            occ = occ.masked_fill(occ < occ_thresh, 0.0)
            alpha = occ * sigma

            x_highres = obj.interp_feats_highres(p)
            ray_dirs_obj = rotate_vectors(obj.obj_from_cam, obj_rays.d)
            xx = torch.cat(
                [
                    expand_dim(x, dim=-2, shape=[x_highres.size(-2)]),
                    x_highres[..., 0:3],
                    rotate_vectors(obj.obj_from_cam, x_highres[..., 3:6]),
                    expand_dim(
                        ray_dirs_obj,
                        dim=-2,
                        shape=[samples_per_ray, x_highres.size(-2)],
                    ),
                    expand_dim(occ, dim=-1, shape=[x_highres.size(-2), 1]),
                ],
                dim=-1,
            )
            radiance = self.radiance(xx).sigmoid()

            total_t[obj_idx, m] = t
            total_occ[obj_idx, m] = occ
            total_alpha[obj_idx, m] = alpha
            total_radiance[obj_idx, m] = radiance

        # (*rays.shape, n_samples, n_objects)
        total_t = total_t.movedim(0, -1)
        total_occ = total_occ.movedim(0, -1)
        total_alpha = total_alpha.movedim(0, -1)

        # (*rays.shape, n_samples, n_objects, 3)
        total_radiance = total_radiance.movedim(0, -1)

        return {
            "t": total_t,
            "alpha": total_alpha,
            "occ": total_occ,
            "radiance": total_radiance,
        }


@attr.s(kw_only=True, eq=False, repr=False)
class VolumeExtractor(nn.Module):
    num_depth_bins: int = attr.ib()
    patch_size: int = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()

        gn = lambda n: GroupNormArgs(num_per_group=n, affine=False)  # noqa: E731

        # Shared 3D-conv feature backbone
        unet_spec = ConvBlobSpec3d(strides=[1, 2, 4, 8], channels=[64, 128, 256, 384])
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=gn(4),
            conv_args=ConvArgs3d(),
        )
        c = 64
        self.embed = nn.Sequential(
            nn.Linear(6, c),
            CollapseCamsBlock(c, 8, 16),
        )
        self.feature_module = nn.Sequential(
            ConvLayer(c, unet_spec.channels[0], spec=conv_spec),
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
        )
        self.squash_out = nn.Sequential(
            Lambda(lambda x_lst: x_lst[0]),
            ConvLayer(unet_spec.channels[0], 1, spec=conv_spec.as_end_spec()),
        )

    def forward(
        self,
        rgbs: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_from_ref: torch.Tensor,
        grid_pts_ref: torch.Tensor,
    ):
        grid_pts_cam = transform_points(
            cam_from_ref[1:, None, None, None, None], grid_pts_ref
        )
        grid_px_cam = project_points_to_pixels(
            grid_pts_cam, intrinsics[1:, None, None, None]
        )

        # shape(n_cams, n_objects, d, h, w, 3)
        grid_pts_viewdir_cam = F.normalize(
            grid_pts_ref[None] - cam_from_ref[1:, None, None, None, None, :3, 3], dim=-1
        )

        # shape(n_cams, n_objects, d, h, w, 6)
        x_in = torch.cat(
            [
                batch_interp2d(rgbs[1:], grid_px_cam),
                grid_pts_viewdir_cam,
            ],
            dim=-1,
        )

        # shape(n_objects, d, h, w, n_cams, 6) -> shape(n_objects, d, h, w, c) -> shape(n_objects, c, d, h, w, )
        x_in = (
            BatchApply(self.embed, dims=3)(x_in.movedim(0, -2)).mean(-2).movedim(-1, 1)
        )

        # shape (n_objects, n_cams * 9, d, h, w) -> shape (n_objects, c, d, h, w)
        x = self.feature_module(x_in)

        # shape (n_objects, c, d, h, w) -> shape (n_objects, d, h, w)
        x_squash = self.squash_out(x).squeeze(1)
        return x, x_squash


@attr.s(kw_only=True, eq=False, repr=False)
class OccupancyDecoder(nn.Module):
    num_depth_bins: int = attr.ib()
    patch_size: int = attr.ib()

    def __attrs_post_init__(self):
        super().__init__()

        gn = lambda n: GroupNormArgs(num_per_group=n, affine=False)  # noqa: E731
        dconv2d = 128

        # 2D-UNet
        unet_channels = [64, 128, 256, 512]
        unet_spec = ConvBlobSpec2d(
            channels=(self.num_depth_bins, *unet_channels),
            strides=(1, *[2**i for i in range(1, len(unet_channels) + 1)]),
        )
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.NORM_NONLIN_CONV,
            nonlin_args=LeakyReluArgs(),
            norm_args=GroupNormArgs(num_groups=4),
        )
        self.unet2d = nn.Sequential(
            UNetSimple(
                in_channels=unet_spec.channels[0],
                unet_spec=unet_spec,
                conv_spec=conv_spec,
            ),
            Lambda(lambda x_lst: x_lst[0]),
        )

        # Final predictor/decoder module
        self.predictor = nn.Sequential(
            ConvLayer(
                self.num_depth_bins,
                dconv2d,
                spec=attr.evolve(conv_spec, norm_args=gn(1)),
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ResidualBlock(
                dconv2d,
                [dconv2d, dconv2d],
                spec=attr.evolve(conv_spec, norm_args=gn(4)),
                mode=ResidualMode.GATED_RESIDUAL,
            ),
            ConvLayer(
                dconv2d,
                self.num_depth_bins,
                spec=ConvLayerSpec(
                    layer_order=LayerOrder.NORM_NONLIN_CONV
                ).as_end_spec(),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet2d(x)
        logits = self.predictor(x).clamp(max=10.0)
        return logits


class CovNerf(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_renderer = ObjectRenderer(input_channels=64, hidden_channels=64)
        self.bg_renderer = BackgroundRenderer()
        self.volume_extractor = VolumeExtractor(num_depth_bins=96, patch_size=48)
        self.occupancy_decoder = OccupancyDecoder(num_depth_bins=96, patch_size=48)

    def render(
        self,
        objects: Sequence[NerfObject],
        *,
        target_intrinsic: torch.Tensor,
        target_world_from_cam: torch.Tensor,
        target_near_plane: float,
        target_far_plane: float,
        target_im_size: torch.Size,
        bg_src_rgbs: torch.Tensor,
        bg_src_intrinsics: torch.Tensor,
        bg_src_world_from_cams: torch.Tensor,
        ray_batch_size: int = 4096,
        n_samples_primary: int = 32,
        n_samples_importance: int = 16,
        n_samples_bg: int = 32,
        verbose: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            rays = Rays.from_camera(
                target_intrinsic, torch.eye(4, device="cuda"), target_im_size
            )
            device = rays.device
            n_objects = len(objects)
            obj_indices = torch.cat(
                [
                    expand_dim(
                        torch.arange(n_objects, device=device),
                        dim=0,
                        shape=[n_samples_primary + n_samples_importance],
                    ).flatten(),
                    torch.full([n_samples_bg], n_objects, device=device),
                ]
            )
            flat_rays = rays.reshape([-1])

            out = defaultdict(list)
            for i in tqdm.trange(
                0, len(flat_rays), ray_batch_size, disable=not verbose
            ):
                ray_batch = flat_rays[i : i + ray_batch_size]
                fg_out = self.object_renderer(
                    ray_batch,
                    objects,
                    n_samples_primary=n_samples_primary,
                    n_samples_importance=n_samples_importance,
                )
                bg_out = self.bg_renderer(
                    F.pad(bg_src_rgbs, (0, 0, 0, 0, 0, 0, 1, 0)),
                    torch.cat([target_intrinsic[None], bg_src_intrinsics]),
                    torch.cat(
                        [target_world_from_cam[None], bg_src_world_from_cams]
                    ).inverse()
                    @ target_world_from_cam,
                    target_near_plane,
                    target_far_plane,
                    ray_batch,
                    n_ray_samples=n_samples_bg,
                )
                flat_t, permute = torch.cat(
                    [fg_out["t"].flatten(-2, -1), bg_out["t"]], dim=-1
                ).sort(-1)
                flat_alpha = torch.cat(
                    [fg_out["alpha"].flatten(-2, -1), bg_out["alpha"]], dim=-1
                ).take_along_dim(permute, dim=-1)
                flat_occ = torch.cat(
                    [fg_out["occ"].flatten(-2, -1), bg_out["alpha"]], dim=-1
                ).take_along_dim(permute, dim=-1)
                flat_radiance = torch.cat(
                    [
                        fg_out["radiance"].movedim(2, 1).flatten(-2, -1),
                        bg_out["radiance"].movedim(2, 1),
                    ],
                    dim=-1,
                ).take_along_dim(permute.unsqueeze(-2), dim=-1)

                flat_obj_indices = obj_indices[permute]

                flat_trans = F.pad(
                    (1 - flat_alpha).log().cumsum(dim=-1).exp(), (1, 0), value=1.0
                )[..., :-1]
                composite_alpha = flat_trans * flat_alpha
                rgb = torch.einsum("...n,...cn->...c", composite_alpha, flat_radiance)
                out["rgb"].append(rgb)

                composite_occ = (
                    flat_occ
                    * F.pad(
                        (1 - flat_occ).log().cumsum(dim=-1).exp(), (1, 0), value=1.0
                    )[..., :-1]
                )
                mask_prob = torch.zeros([len(ray_batch), n_objects + 1], device=device)
                torch.scatter_add(
                    mask_prob, dim=1, index=flat_obj_indices, src=composite_occ
                )
                out["segm"].append(mask_prob[..., :-1])

                hit_dist = flat_t.take_along_dim(
                    composite_occ.argmax(-1).unsqueeze(-1), dim=-1
                ).squeeze(-1)
                depth = ray_batch.get_points(hit_dist)[..., 2]
                out["depth"].append(depth)

        return {
            "rgb": torch.cat(out["rgb"]).view(*target_im_size, 3),
            "depth": torch.cat(out["depth"]).view(*target_im_size),
            "segm": torch.cat(out["segm"])
            .view(*target_im_size, n_objects)
            .movedim(-1, 0),
        }

    def extract_objects(
        self,
        rgbs: torch.Tensor,
        intrinsics: torch.Tensor,
        world_from_cam: torch.Tensor,
        frustums: Frustum,
        *,
        occupancy_thresh: float = 0.7,
    ) -> Sequence[NerfObject]:
        device = rgbs.device
        frustum_voxel_shape = torch.tensor(
            [
                self.volume_extractor.num_depth_bins,
                self.volume_extractor.patch_size,
                self.volume_extractor.patch_size,
            ],
            device=rgbs.device,
            dtype=torch.int64,
        )
        grid_pts_ref = frustums.to_grid(frustum_voxel_shape, alignment="endpoint")

        cam_from_ref = world_from_cam.inverse() @ world_from_cam[0]
        feature_vols, feature_patches = self.volume_extractor(
            rgbs, intrinsics, cam_from_ref, grid_pts_ref
        )
        voxel_probs = self.occupancy_decoder(feature_patches)

        pts_lst = []
        for i in range(len(voxel_probs)):
            obj_pts_ref = grid_pts_ref[i, voxel_probs[i] > occupancy_thresh]
            pts_lst.append(obj_pts_ref)

        poses, dimensions = get_bbox3d_from_occupancy(pts_lst)

        s = torch.linspace(-0.5, 0.5, 64, device=device)
        s_highres = torch.linspace(-0.5, 0.5, 128, device=device)

        # shape (d, h, w, 3)
        pts_normed = torch.stack(torch.meshgrid(s, s, s, indexing="ij"), dim=-1).flip(
            -1
        )
        pts_obj = pts_normed * dimensions[:, None, None, None]
        # shape (n_objects, d, h, w, 3)
        pts_cam = transform_points(poses[:, None, None, None], pts_obj)

        objects = []
        for i in range(len(frustums)):
            uvz = frustums[i].normalize_coords(pts_cam[i])
            dhw = uvz.flip(-1) * frustum_voxel_shape.sub(1)
            voxel_probs_resampled = interp3d(voxel_probs[i, None], dhw).squeeze(-1)
            feature_vol_resampled = interp3d(feature_vols[0][i], dhw).movedim(-1, 0)

            grid_highres_normed = torch.stack(
                torch.meshgrid(s_highres, s_highres, s_highres, indexing="ij"), dim=-1
            )
            grid_src = transform_points(
                poses[i, None, None, None], grid_highres_normed * dimensions[i]
            )
            grid_px_src = project_points_to_pixels(grid_src, intrinsics[:, None, None])
            ray_dirs = F.normalize(
                grid_src - cam_from_ref[:, None, None, None, :3, 3], p=2, dim=-1
            )
            ray_dirs = rotate_vectors(poses[i], ray_dirs)
            vol_highres = torch.cat(
                [batch_interp2d(rgbs, grid_px_src), ray_dirs], dim=-1
            ).movedim(-1, 1)

            obj = NerfObject(
                pose=poses[i],
                dimension=dimensions[i],
                voxel_probs=voxel_probs_resampled,
                feature_vol=feature_vol_resampled,
                feature_vol_highres=vol_highres,
            )
            objects.append(obj)

        return tuple(objects)
