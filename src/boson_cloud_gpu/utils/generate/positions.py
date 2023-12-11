# ADDLICENSE
# %%
import cupy
from cupy.typing import NDArray

import sys

sys.path.append("../../")
import settings

# import numpy as cupy

ROUND = [
    "sphere",
    "disk",
    "ellipse",
    "ellipsoid",
]

SUPPORTED_SHAPES = ROUND


def is_supported(item: str, supported_list: list[str]):
    return item in supported_list


class Positions:
    def __init__(
        self,
        n_points_per_cluster: int,
        n_clusters: int = 1,
        cluster_3D_sizes: list[float, float, float] = [1, 1, 1],  # parsec
        cluster_shape: str = "sphere",
        empty_shape: bool = False,
        # velocities: bool = True,  # TODO not implemented
    ) -> None:
        self._n_points_per_cluster = n_points_per_cluster
        self._n_clusters = n_clusters
        self._cluster_shape = cluster_shape
        self._empty_shape = empty_shape
        self._cluster_3D_sizes = cupy.array(cluster_3D_sizes)
        # self._distribution_name = distribution_name
        # self._velocities = velocities

        self.shape = [self._n_clusters, self._n_points_per_cluster]

        assert is_supported(cluster_shape, SUPPORTED_SHAPES)
        assert isinstance(n_points_per_cluster, int)
        assert isinstance(n_clusters, int)

    def gaussian(self):
        if self._empty_shape:
            raise ValueError("Gaussian distributed points on surface not defined")
        positions = cupy.random.multivariate_normal(
            [0, 0, 0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            size=self.shape,
        )
        normalized_positions = positions / cupy.abs(positions).max()
        rescaled_positions = (
            normalized_positions * self._cluster_3D_sizes[None, None, :] / 2
        )
        return rescaled_positions.astype(settings.GENERAL["PRECISION"])

    def uniform(self):
        if self._empty_shape:
            radius = 1
        else:
            cube_radius = cupy.random.uniform(0, 1, self.shape).astype(
                settings.GENERAL["PRECISION"]
            )
            radius = cupy.power(cube_radius, 1 / 3)
        phi = cupy.random.uniform(0, 2 * cupy.pi, self.shape).astype(
            settings.GENERAL["PRECISION"]
        )
        costheta = cupy.random.uniform(-1, 1, self.shape).astype(
            settings.GENERAL["PRECISION"]
        )
        theta = cupy.arccos(costheta)

        x = radius * cupy.sin(theta) * cupy.cos(phi)
        y = radius * cupy.sin(theta) * cupy.sin(phi)
        z = radius * costheta

        positions = cupy.array([x, y, z], dtype=settings.GENERAL["PRECISION"])
        rescaled_positions = positions * self._cluster_3D_sizes[:, None, None] / 2
        reshaped_positions = cupy.einsum("ijk -> jki", rescaled_positions)

        return reshaped_positions.astype(settings.GENERAL["PRECISION"])


def merge_position_clusters(cluster_1: NDArray, cluster_2: NDArray):
    cluster_1_pos_array = cupy.reshape(
        cluster_1,
        (
            cluster_1.shape[0] * cluster_1.shape[1],
            cluster_1.shape[2],
        ),
    )
    cluster_2_pos_array = cupy.reshape(
        cluster_2,
        (
            cluster_2.shape[0] * cluster_2.shape[1],
            cluster_2.shape[2],
        ),
    )

    return cupy.concatenate(
        [cluster_1_pos_array, cluster_2_pos_array],
        axis=0,
        dtype=settings.GENERAL["PRECISION"],
    )
