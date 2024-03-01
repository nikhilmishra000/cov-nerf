from typing import Any, Optional, Sequence, Union

import attr
import cv2
import ipywidgets
import matplotlib
import numpy as np
import pythreejs
import torch
from IPython.display import display
from matplotlib import pyplot as plt

from torch_utils import to_np

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Union[int, float, Sequence[Any]]]]


def _to_hwc(x):
    if x.shape[0] == 3:
        return x.transpose(1, 2, 0)
    elif x.shape[2] == 3:
        return x
    else:
        raise ValueError(x.shape)


def plot_rgb(rgb, ax=None):
    rgb = _to_hwc(to_np(rgb))
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb.astype(np.float32)

    if ax is None:
        plt.imshow(rgb)
    else:
        ax.imshow(rgb)


def plot_mask(
    mask: ArrayLike,
    c: Optional[Union[str, ArrayLike]] = None,
    edgecolor: Optional[Union[str, ArrayLike]] = None,
    alpha: float = 0.7,
    linewidth: float = 0.5,
    ax: Optional[Any] = None,
) -> Any:
    """Overlay the given mask on the given ax (or the current plt.gca()).

    Parameters
    ----------
    mask: array of shape H x W, of dtype bool, uint8 or floating.
        If the mask dtype is bool or uint8, the mask corresponds to all the non-zero elements.
        If it's a float, we plot a colormap wherever the entries are finite.
    c: Optional[Union[str, Sequence[int, int, int]]]
        Color to use for the mask, specified as a string, or as an RGB triplet (see matplotly docs).
    edgecolor: Optional[Union[str, Sequence[int, int, int]]]
        Color to use for the edge of the mask (same type as c above). If not provided, edges are not plotted.
    alpha: float
        Transparency of the mask.
    linewidth: flot
        Width of the mask edge, if plotted.
    ax: Optional[Any]
        If provided, plot the mask on this axis. Otherwise, they are obtained with plt.gca().

    Returns
    -------
        The mask image, in case we need it for plotting colorbars.
    """
    if ax is None:
        ax = plt.gca()
    if c is None:
        c = np.random.uniform(0.2, 1.0, size=3)
    elif isinstance(c, str):
        c = matplotlib.colors.to_rgb(c)
    else:
        assert len(c) == 3

    mask = to_np(mask)
    h, w = mask.shape
    colored_mask = np.zeros([h, w, 4], dtype=np.float32)
    if mask.dtype == np.bool or mask.dtype == np.uint8:
        colored_mask[mask.astype(np.bool_)] = (*c, alpha)
    elif np.issubdtype(mask.dtype, np.floating):
        m = np.isfinite(mask)
        colored_mask[m] = plt.get_cmap()(mask[m])
        colored_mask[..., -1] = m * alpha
    else:
        raise NotImplementedError(mask.dtype)

    im = ax.imshow(colored_mask)
    if edgecolor is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
        )
        for pts in contours:
            ax.plot(*pts.reshape(-1, 2).T, c=edgecolor, linewidth=linewidth)
    return im


@attr.s(frozen=True, kw_only=True, repr=False)
class SceneBuilder:
    scene: pythreejs.Scene = attr.ib()
    renderer: pythreejs.Renderer = attr.ib()
    flip_xz: bool = attr.ib()

    # TODO: these should get handled by the caller who wants them
    extra_widgets: list[ipywidgets.Widget] = attr.ib(factory=list)

    @classmethod
    def create(
        cls,
        height: int,
        width: int,
        background_color: str = "black",
        flip_xz: bool = True,
    ) -> "SceneBuilder":
        """Create an empty scene.

        This will include a camera, orbit controls, and ambient lighting, but no geometry.
        """
        camera = pythreejs.PerspectiveCamera(
            fov=90, aspect=width / height, position=[1, 0, 0], up=[0, 0, 1]
        )
        orbit_controls = pythreejs.OrbitControls(controlling=camera)
        scene = pythreejs.Scene(
            children=[camera, orbit_controls, pythreejs.AmbientLight(color="#FFFFFF")],
            background=background_color,
        )
        renderer = pythreejs.Renderer(
            scene=scene,
            camera=camera,
            controls=[orbit_controls],
            width=width,
            height=height,
        )
        return cls(scene=scene, renderer=renderer, flip_xz=flip_xz)

    @classmethod
    def from_point_map(
        cls,
        xyz: ArrayLike,
        rgb: ArrayLike,
        size: float = 0.001,
        flip_xz: bool = True,
        render_size: tuple[int, int] = (800, 600),
    ) -> "SceneBuilder":
        """Create a new scene from a point map and image.

        Parameters
        ----------
        xyz : ArrayLike
            Points in 3d. Shape(h,w,3) or (3.h,w).
        rgb : ArrayLike
            The image corresponding to `xyz`. Should have the same shape.
        size : float, optional
            The size of the plotted points.
        max_pts : Optional[int], optional
            If given, downsample the point cloud so that we have at most this many points.
        flip_xz : bool, optional
            It is often convenient to flip these axes so that the initial viewpoint is more closely aligned to 2D image visualizations.
            TODO (nikhil): do we actually need this? should be fixable by positioning the camera better.
        render_size : Tuple[int, int]
            The size of the rendered visualization in pixels (width, height).
        """
        xyz = _to_hwc(to_np(xyz))  # shape(h,w,3), numpy array
        rgb = _to_hwc(to_np(rgb))  # shape(h,w,3), numpy array
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        assert xyz.shape == rgb.shape, (xyz.shape, rgb.shape)
        xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
        builder = cls.create(
            height=render_size[1], width=render_size[0], flip_xz=flip_xz
        )
        builder.add_points(points=xyz, color=rgb, size=size)

        # Update the camera pose to something sensible
        center, maximums = xyz.mean(0), xyz.max(0)
        if builder.flip_xz:
            center, maximums = center * [-1, 1, -1], maximums * [-1, 1, -1]

        builder.renderer.camera.near = 1e-3
        builder.renderer.camera.position = tuple(
            center + [0, abs(maximums[1]), abs(maximums[2]) * 1.5]
        )
        builder.renderer.camera.lookAt(tuple(center))

        # Update the orbit controls so the center of rotation is at the center of the point map.
        for controls in builder.renderer.controls:
            if isinstance(controls, pythreejs.OrbitControls):
                controls.target = tuple(center)
            else:
                raise NotImplementedError(type(controls))

        # Add a slider to control the point size.
        # Ideally this should happen outside of SceneBuilder
        points_geom = builder.scene.children[-1]
        assert isinstance(points_geom, pythreejs.Points), type(points_geom)
        slider = ipywidgets.FloatSlider(
            value=size, min=0.0, max=size * 10, step=size / 100
        )
        ipywidgets.jslink((slider, "value"), (points_geom.material, "size"))
        builder.extra_widgets.append(
            ipywidgets.HBox([ipywidgets.Label("Point size:"), slider])
        )
        return builder

    def add_line_segments(
        self,
        points: ArrayLike,
        color: Union[str, ArrayLike] = "red",
        point_size: float = 0.001,
        thickness: float = 1.0,
    ) -> "SceneBuilder":
        """Add line segments connecting a list of points to the scene.

        Parameters
        ----------
        points : ArrayLike
            Shape (n, 3).
            Lines will be drawn using `add_lines()` between `points[i]` and `points[i + 1]`.
            Points will be drawn at each point using `add_points()`.
        color : Union[str, ArrayLike]
        point_size : float
            Passed as `size` to `add_points()`.
        thickness : float, optional
            Passed as `thickness` to `add_lines()`.

        Returns
        -------
        SceneBuilder
        """
        points = to_np(points)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("Expected points of shape (-1, 3)!")

        self.add_lines(points[:-1], points[1:], thickness=thickness, color=color)
        self.add_points(points, color=color, size=point_size)
        return self

    def add_points(
        self,
        points: ArrayLike,
        color: Union[str, ArrayLike] = "red",
        size: float = 0.001,
    ) -> "SceneBuilder":
        """Add points to the scene.

        Parameters
        ----------
        points : np.ndarray
            Points in 3D. Shape(..., 3).
        color : Union[str, np.ndarray]
            Plot `points` using this color. It can be:
              - A named color that `matplotlib.colors` can understand (e.g. "red" or "blue").
              - A hex code (e.g. "#ff0033").
              - A single color expressed as a sequence of 3 floats.
              - An array of the same shape as `points`.
        size : float
        """
        points = to_np(points)
        if self.flip_xz:
            points = points * [-1, 1, -1]

        if isinstance(color, str):
            color = matplotlib.colors.to_rgb(color)
        color = np.broadcast_to(to_np(color), points.shape)

        mask = np.isfinite(points).all(-1)
        points = points[mask]
        color = color[mask]

        geometry = pythreejs.BufferGeometry(
            attributes=dict(
                position=pythreejs.BufferAttribute(
                    points.astype(np.float32), normalized=False
                ),
                color=pythreejs.BufferAttribute(color.astype(np.float32)),
            )
        )
        material = pythreejs.PointsMaterial(vertexColors="VertexColors", size=size)
        self.scene.add(pythreejs.Points(geometry=geometry, material=material))
        return self

    def show(self) -> None:
        display(ipywidgets.VBox([self.renderer, *self.extra_widgets]))
