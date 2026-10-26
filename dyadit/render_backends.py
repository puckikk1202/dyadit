"""Fixed-camera offscreen rendering, with a CPU rasterizer fallback."""

import ctypes.util
import os
from dataclasses import dataclass
from itertools import product

import numpy as np


def look_at(eye, target):
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 1, 0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, up, -forward))
    pose[:3, 3] = eye
    return pose


@dataclass
class Camera:
    pose: np.ndarray
    xmag: float
    ymag: float
    width: int
    height: int

    @classmethod
    def fit(cls, bounds, settings):
        center = bounds.mean(axis=0)
        azimuth, elevation = np.deg2rad(
            [settings.camera_azimuth, settings.camera_elevation]
        )
        direction = np.array(
            [
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
                np.cos(azimuth) * np.cos(elevation),
            ]
        )
        pose = look_at(center + direction * 6, center)
        corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
        projected = (corners - pose[:3, 3]) @ pose[:3, :3]
        aspect = settings.width / settings.height
        ymag = (
            max(np.abs(projected[:, 1]).max(), np.abs(projected[:, 0]).max() / aspect)
            * 1.12
        )
        return cls(pose, ymag * aspect, ymag, settings.width, settings.height)

    def project(self, vertices):
        points = (vertices - self.pose[:3, 3]) @ self.pose[:3, :3]
        pixels = np.column_stack(
            (
                (points[:, 0] / (2 * self.xmag) + 0.5) * self.width,
                (0.5 - points[:, 1] / (2 * self.ymag)) * self.height,
            )
        )
        return pixels, -points[:, 2]


class SoftwareRenderer:
    """CPU triangle rendering; no OpenGL context or display server is needed."""

    backend = "software"

    def __init__(self, settings, bounds, faces, colors):
        import cv2

        self.cv2 = cv2
        self.camera = Camera.fit(bounds, settings)
        self.width, self.height = settings.width, settings.height
        self.faces, self.colors = faces, colors
        self.light = np.array([-0.3, 0.8, 1.2])
        self.light /= np.linalg.norm(self.light)

    def render(self, meshes):
        cv2 = self.cv2
        image = np.full((self.height, self.width, 3), (245, 247, 250), dtype=np.uint8)
        floor = np.array([[-4, 0, -4], [4, 0, -4], [4, 0, 4], [-4, 0, 4]], dtype=float)
        pixels, _ = self.camera.project(floor)
        cv2.fillConvexPoly(
            image,
            np.rint(pixels).astype(np.int32),
            (229, 233, 238),
            lineType=cv2.LINE_AA,
        )
        for value in np.arange(-4, 4.01, 0.5):
            for line in (
                np.array([[value, 0, -4], [value, 0, 4]]),
                np.array([[-4, 0, value], [4, 0, value]]),
            ):
                points, _ = self.camera.project(line)
                cv2.line(
                    image,
                    tuple(np.rint(points[0]).astype(int)),
                    tuple(np.rint(points[1]).astype(int)),
                    (218, 223, 230),
                    1,
                    cv2.LINE_AA,
                )
        polygons, depths, colors = [], [], []
        for vertices, faces, color in zip(meshes, self.faces, self.colors):
            triangles = vertices[faces]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            normals /= np.maximum(
                np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12
            )
            visible = normals @ self.camera.pose[:3, 2] > 0
            points, depth = self.camera.project(vertices)
            brightness = 0.48 + 0.52 * np.maximum(normals @ self.light, 0)
            polygons.append(np.rint(points[faces[visible]]).astype(np.int32))
            depths.append(depth[faces[visible]].mean(axis=1))
            colors.append(
                np.clip(
                    np.asarray(color) * brightness[visible, None] * 255, 0, 255
                ).astype(np.uint8)
            )
        polygons, depths, colors = map(np.concatenate, (polygons, depths, colors))
        for index in np.argsort(depths)[::-1]:
            cv2.fillConvexPoly(
                image,
                polygons[index],
                tuple(int(x) for x in colors[index]),
                lineType=cv2.LINE_AA,
            )
        return image

    def close(self):
        pass


class PyrenderRenderer:
    def __init__(self, settings, bounds, faces, colors, platform):
        previous = os.environ.get("PYOPENGL_PLATFORM")
        if previous and previous != platform:
            raise RuntimeError(
                f"PYOPENGL_PLATFORM={previous} conflicts with requested backend {platform}"
            )
        os.environ["PYOPENGL_PLATFORM"] = platform
        import pyrender
        import trimesh

        self.pyrender, self.trimesh = pyrender, trimesh
        self.backend = platform
        self.camera = Camera.fit(bounds, settings)
        self.faces, self.colors = faces, colors
        self.nodes = []
        self.scene = pyrender.Scene(
            bg_color=[0.96, 0.97, 0.99, 1], ambient_light=[0.35, 0.35, 0.35]
        )
        floor = trimesh.creation.box(extents=(8, 0.02, 8))
        floor_pose = np.eye(4)
        floor_pose[1, 3] = -0.01
        floor_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.86, 0.88, 0.91, 1], metallicFactor=0, roughnessFactor=0.9
        )
        self.scene.add(
            pyrender.Mesh.from_trimesh(floor, material=floor_material), pose=floor_pose
        )
        camera = pyrender.OrthographicCamera(
            xmag=self.camera.xmag, ymag=self.camera.ymag, znear=0.01, zfar=100
        )
        self.scene.add(camera, pose=self.camera.pose)
        for position, intensity in (((-3, 5, 4), 2.5), ((4, 3, 1), 1.2)):
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            self.scene.add(light, pose=look_at(position, (0, 1, 0)))
        self.renderer = pyrender.OffscreenRenderer(settings.width, settings.height)

    def render(self, meshes):
        for node in self.nodes:
            self.scene.remove_node(node)
        self.nodes = []
        for vertices, faces, color in zip(meshes, self.faces, self.colors):
            material = self.pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*color, 1], metallicFactor=0, roughnessFactor=0.65
            )
            mesh = self.trimesh.Trimesh(vertices, faces, process=False)
            node = self.scene.add(
                self.pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
            )
            self.nodes.append(node)
        image, _ = self.renderer.render(
            self.scene, flags=self.pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        )
        return image[:, :, :3]

    def close(self):
        self.renderer.delete()


def make_renderer(settings, bounds, faces, colors):
    if settings.backend == "software":
        return SoftwareRenderer(settings, bounds, faces, colors)
    platform = settings.backend
    if platform == "auto":
        import torch

        platform = os.environ.get("PYOPENGL_PLATFORM") or (
            "egl" if torch.cuda.is_available() else "osmesa"
        )
        if platform == "osmesa" and not ctypes.util.find_library("OSMesa"):
            return SoftwareRenderer(settings, bounds, faces, colors)
    try:
        return PyrenderRenderer(settings, bounds, faces, colors, platform)
    except Exception as error:
        if settings.backend != "auto":
            raise RuntimeError(
                f"{platform} rendering failed: {error}. Try --backend software."
            ) from error
        print(
            f"{platform} unavailable ({error}); using CPU software rendering.",
            flush=True,
        )
        return SoftwareRenderer(settings, bounds, faces, colors)
