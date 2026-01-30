
r"""
Ash's OBJ Viewer (PyQt5)

OBJ viewer + turntable renderer for when you want a rotating product shot
without starting a Blender pilgrimage.

New in this version:
- Light brightness slider (scales light contribution)
- Light diffuse slider (controls how matte-bright it is)
- Background colour picker (also affects renders because render == framebuffer grab)

Everything else:
- Dark navy + orange theme
- Scrollable control panel
- Faces / Edges / Wireframe-only
- Edge alpha slider
- Render clean (disable edges for export only)
- Smooth vs Flat shading toggle
- Two-tone studio preset
- Lighting modes: headlight (camera) or fixed (world)
- Face colour picker + light colour picker
- Grid + axis toggles (also in render)

Compatibility notes:
- Avoids GLMeshItem.meshData() because some pyqtgraph versions don’t have it.
- Prefers shader="color" or "vertexColor" for baked lighting, falls back to "shaded".

Dependencies:
    python -m pip install PyQt5 pyqtgraph PyOpenGL numpy trimesh imageio imageio-ffmpeg Pillow

Run:
    python main.py
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

import trimesh
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import imageio.v2 as imageio


Axis = Literal["x", "y", "z"]
LightMode = Literal["Headlight (camera)", "Fixed (world)"]


# ----------------------------
# Theme (navy + orange)
# ----------------------------

THEME = {
    "accent": "#ff8c00",
    "bg": "#0b1220",
    "panel": "#0f1a2b",
    "panel2": "#0c1626",
    "input": "#111f33",
    "input2": "#0f1b2e",
    "border": "#22314d",
    "text": "#e8eefc",
    "muted": "#9fb0d0",
}


def apply_qt_theme(app: QtWidgets.QApplication) -> None:
    """My jazzy colour scheme, not a corporate printer driver."""
    qss = f"""
    * {{
        font-family: "Segoe UI";
        font-size: 10.5pt;
        color: {THEME["text"]};
    }}

    QMainWindow, QWidget {{
        background: {THEME["bg"]};
    }}

    QScrollArea {{
        background: {THEME["bg"]};
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: {THEME["bg"]};
    }}

    QFrame#LeftPanel {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        border-radius: 12px;
    }}

    QGroupBox {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        border-radius: 12px;
        margin-top: 16px;
        padding: 10px;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 8px;
        left: 10px;
        top: -10px;
        background: {THEME["panel2"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        color: {THEME["accent"]};
        font-weight: 700;
    }}

    QLabel {{
        background: transparent;
    }}

    QLineEdit {{
        background: {THEME["input"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        padding: 7px 10px;
        selection-background-color: {THEME["accent"]};
        selection-color: #000000;
    }}
    QLineEdit:focus {{
        border: 1px solid {THEME["accent"]};
    }}

    QSpinBox, QDoubleSpinBox {{
        background: {THEME["input"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        padding: 6px 8px;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {THEME["accent"]};
    }}

    QComboBox {{
        background: {THEME["input"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QComboBox:focus {{
        border: 1px solid {THEME["accent"]};
    }}
    QComboBox QAbstractItemView {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        selection-background-color: {THEME["accent"]};
        selection-color: #000000;
        outline: 0;
    }}

    QPushButton {{
        background: {THEME["input2"]};
        border: 1px solid {THEME["border"]};
        border-radius: 10px;
        padding: 8px 10px;
        font-weight: 600;
        min-height: 30px;
    }}
    QPushButton:hover {{
        border: 1px solid {THEME["accent"]};
        background: #132543;
    }}
    QPushButton:pressed {{
        background: #0f2440;
        border: 1px solid {THEME["accent"]};
    }}

    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid {THEME["border"]};
        background: {THEME["input"]};
    }}
    QCheckBox::indicator:hover {{
        border: 1px solid {THEME["accent"]};
    }}
    QCheckBox::indicator:checked {{
        background: {THEME["accent"]};
        border: 1px solid {THEME["accent"]};
    }}

    QSlider::groove:horizontal {{
        height: 8px;
        background: {THEME["input"]};
        border: 1px solid {THEME["border"]};
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
        background: {THEME["accent"]};
        border: 1px solid {THEME["accent"]};
    }}

    QProgressBar {{
        background: {THEME["input"]};
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        height: 18px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {THEME["accent"]};
        border-radius: 8px;
    }}
    """
    app.setStyle("Fusion")
    app.setStyleSheet(qss)


# ----------------------------
# Maths helpers
# ----------------------------

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def rotation_matrix(axis: Axis, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]], dtype=np.float32)
    if axis == "y":
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=np.float32)

    raise ValueError(f"Unknown axis: {axis}")


# ----------------------------
# Loading helpers
# ----------------------------

def resolve_obj_path(obj_arg: str) -> str:
    obj_arg = os.path.abspath(os.path.expanduser(obj_arg))

    if os.path.isfile(obj_arg):
        if obj_arg.lower().endswith(".obj"):
            return obj_arg
        raise ValueError(f"Selected file is not an .obj: {obj_arg}")

    if os.path.isdir(obj_arg):
        candidates = [
            os.path.join(obj_arg, f)
            for f in os.listdir(obj_arg)
            if f.lower().endswith(".obj")
        ]
        if not candidates:
            raise ValueError(f"Folder contains no .obj files: {obj_arg}")
        if len(candidates) > 1:
            names = "\n".join("  - " + os.path.basename(c) for c in candidates)
            raise ValueError(f"Folder contains multiple .obj files, pick one:\n{names}")
        return os.path.abspath(candidates[0])

    raise ValueError(f"Path is not a file or folder: {obj_arg}")


def load_trimesh_baked(obj_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(obj_path, force="scene")

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            geom = loaded.to_geometry()
            if isinstance(geom, trimesh.Trimesh):
                mesh = geom
            else:
                items = list(geom)  # type: ignore
                if not items:
                    raise ValueError("No mesh geometry found in the OBJ scene.")
                mesh = trimesh.util.concatenate(items)
        else:
            dumped = loaded.dump(concatenate=True)
            if isinstance(dumped, list):
                if not dumped:
                    raise ValueError("No mesh geometry found in the OBJ scene.")
                mesh = trimesh.util.concatenate(dumped)
            else:
                mesh = dumped
    else:
        raise TypeError(f"Unsupported type from trimesh.load: {type(loaded)}")

    if mesh.vertices is None or len(mesh.vertices) == 0 or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Loaded mesh is empty (no vertices/faces).")

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target_diag: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)

    diag = float(np.linalg.norm(m.bounding_box.extents))
    if diag > 0:
        m.apply_scale(target_diag / diag)

    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int32)
    return verts, faces


def compute_radius(verts: np.ndarray) -> float:
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    ext = bmax - bmin
    r = float(np.linalg.norm(ext) * 0.5)
    return max(r, 1e-3)


def compute_vertex_normals_smooth(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)

    n = np.zeros_like(verts, dtype=np.float32)
    np.add.at(n, faces[:, 0], fn)
    np.add.at(n, faces[:, 1], fn)
    np.add.at(n, faces[:, 2], fn)
    return _normalize(n)


def make_flat_shaded_mesh(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vf = verts[faces].reshape(-1, 3).astype(np.float32)
    ff = np.arange(len(vf), dtype=np.int32).reshape(-1, 3)

    v0 = vf[ff[:, 0]]
    v1 = vf[ff[:, 1]]
    v2 = vf[ff[:, 2]]
    fn = _normalize(np.cross(v1 - v0, v2 - v0))
    nf = np.repeat(fn, 3, axis=0).astype(np.float32)
    return vf, ff, nf


# ----------------------------
# Lighting (baked vertex colors)
# ----------------------------

@dataclass
class LightingParams:
    base_rgb: Tuple[float, float, float] = (0.92, 0.92, 0.98)  # model face color
    light_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)    # light tint
    ambient: float = 0.20
    diffuse: float = 0.95
    specular: float = 0.18
    shininess: float = 48.0
    brightness: float = 1.0  # NEW: scales (diffuse+specular) contribution


def bake_vertex_colors(
    verts_world: np.ndarray,
    normals_world: np.ndarray,
    light_pos_world: np.ndarray,
    cam_pos_world: np.ndarray,
    params: LightingParams,
) -> np.ndarray:
    L = _normalize(light_pos_world[None, :] - verts_world)
    N = _normalize(normals_world)

    ndotl = np.clip(np.sum(N * L, axis=1), 0.0, 1.0)

    V = _normalize(cam_pos_world[None, :] - verts_world)
    H = _normalize(L + V)
    ndoth = np.clip(np.sum(N * H, axis=1), 0.0, 1.0)
    spec = np.power(ndoth, params.shininess)

    base = np.array(params.base_rgb, dtype=np.float32)[None, :]
    light = np.array(params.light_rgb, dtype=np.float32)[None, :]

    # Ambient is "environment light". Diffuse/specular are "light" and get brightness + tint.
    light_scale = float(max(params.brightness, 0.0))
    col = (
        base * params.ambient
        + (base * (params.diffuse * ndotl[:, None]) * light) * light_scale
        + (params.specular * spec[:, None] * light) * light_scale
    )

    col = np.clip(col, 0.0, 1.0).astype(np.float32)
    alpha = np.ones((col.shape[0], 1), dtype=np.float32)
    return np.concatenate([col, alpha], axis=1)


# ----------------------------
# Render config
# ----------------------------

@dataclass
class RenderSettings:
    fps: int
    seconds: float
    degrees: float
    axis: Axis
    out_mp4: str
    out_gif: Optional[str]


# ----------------------------
# GLMeshItem helpers
# ----------------------------

def _mesh_set_opt(mesh_item: gl.GLMeshItem, key: str, value) -> None:
    mesh_item.opts[key] = value
    mesh_item.update()


def _mesh_set_edge_color(mesh_item: gl.GLMeshItem, rgba: Tuple[float, float, float, float]) -> None:
    if hasattr(mesh_item, "setEdgeColor"):
        try:
            mesh_item.setEdgeColor(rgba)  # type: ignore
            return
        except Exception:
            pass
    _mesh_set_opt(mesh_item, "edgeColor", rgba)


def choose_shader(preferred: Tuple[str, ...] = ("color", "vertexColor", "shaded")) -> str:
    try:
        from pyqtgraph.opengl import shaders  # type: ignore
        names = getattr(getattr(shaders, "ShaderProgram", None), "names", None)
        if isinstance(names, dict):
            for s in preferred:
                if s in names:
                    return s
    except Exception:
        pass
    return preferred[0]


# ----------------------------
# UI
# ----------------------------

class TurntableWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ash's OBJ Viewer")

        # Raw mesh
        self._verts_norm: Optional[np.ndarray] = None
        self._faces_norm: Optional[np.ndarray] = None

        # Display mesh
        self._base_verts: Optional[np.ndarray] = None
        self._base_faces: Optional[np.ndarray] = None
        self._base_normals: Optional[np.ndarray] = None

        # Current verts
        self._cur_verts: Optional[np.ndarray] = None

        self.mesh_item: Optional[gl.GLMeshItem] = None
        self.bounds_radius: float = 1.0
        self._ui_busy: bool = False

        # Default colours
        self.model_color_rgb: Tuple[float, float, float] = (0.92, 0.92, 0.98)
        self.light_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)

        # Background colour (NEW)
        self.bg_rgb_255: Tuple[int, int, int] = (11, 18, 32)  # navy-ish default
        self.bg_studio_rgb_255: Tuple[int, int, int] = (14, 12, 20)

        # Edge colours
        self._edge_default_rgb = (1.0, 0.55, 0.0)
        self._edge_studio_rgb = (0.80, 0.70, 0.55)

        # Lighting state
        self.light_mode: LightMode = "Headlight (camera)"
        self.light_az_deg: float = 45.0
        self.light_el_deg: float = 35.0
        self.light_dist: float = 8.0

        # Light behaviour (NEW sliders)
        self.light_brightness: float = 1.0
        self.light_diffuse: float = 0.95

        # Base lighting params (studio will override a few)
        self.lighting_params = LightingParams(
            base_rgb=self.model_color_rgb,
            light_rgb=self.light_color_rgb,
            ambient=0.20,
            diffuse=self.light_diffuse,
            specular=0.18,
            shininess=48.0,
            brightness=self.light_brightness,
        )
        self._studio_params = LightingParams(
            base_rgb=self.model_color_rgb,
            light_rgb=self.light_color_rgb,
            ambient=0.28,
            diffuse=0.85,
            specular=0.22,
            shininess=72.0,
            brightness=self.light_brightness,
        )

        self._shader_name = choose_shader()

        # ----------------------------
        # Layout
        # ----------------------------
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_layout = QtWidgets.QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left scroll panel
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setMinimumWidth(540)
        main_layout.addWidget(self.scroll, 0)

        self.left_panel = QtWidgets.QFrame()
        self.left_panel.setObjectName("LeftPanel")
        self.scroll.setWidget(self.left_panel)

        left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(12)

        # Viewport
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view.setBackgroundColor(self.bg_rgb_255)
        main_layout.addWidget(self.view, 1)

        # Grid + axis
        self.grid = gl.GLGridItem()
        self.grid.setSize(10, 10)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -1.0)
        self.view.addItem(self.grid)

        self.axis_item = gl.GLAxisItem()
        self.axis_item.setSize(2, 2, 2)
        self.view.addItem(self.axis_item)

        # ----------------------------
        # View
        # ----------------------------
        grp_view = QtWidgets.QGroupBox("View")
        left_layout.addWidget(grp_view)
        vf = QtWidgets.QFormLayout(grp_view)
        vf.setVerticalSpacing(10)

        self.chk_show_grid = QtWidgets.QCheckBox("Show grid")
        self.chk_show_grid.setChecked(True)
        self.chk_show_grid.stateChanged.connect(self.apply_view_toggles)

        self.chk_show_axis = QtWidgets.QCheckBox("Show axis")
        self.chk_show_axis.setChecked(True)
        self.chk_show_axis.stateChanged.connect(self.apply_view_toggles)

        self.btn_bg_color = QtWidgets.QPushButton("Pick background colour…")
        self.btn_bg_color.clicked.connect(self.pick_background_color)
        self._apply_color_button_style(self.btn_bg_color, self._rgb255_to_float(self.bg_rgb_255))

        vf.addRow(self.chk_show_grid)
        vf.addRow(self.chk_show_axis)
        vf.addRow(self.btn_bg_color)

        # ----------------------------
        # Model + Output
        # ----------------------------
        grp_io = QtWidgets.QGroupBox("Model + Output")
        left_layout.addWidget(grp_io)
        io = QtWidgets.QFormLayout(grp_io)
        io.setHorizontalSpacing(10)
        io.setVerticalSpacing(10)

        self.btn_load = QtWidgets.QPushButton("Load OBJ…")
        self.btn_load.clicked.connect(self.on_load_obj)
        io.addRow(self.btn_load)

        self.lbl_obj = QtWidgets.QLabel("(no model loaded)")
        self.lbl_obj.setWordWrap(True)
        io.addRow("Loaded:", self.lbl_obj)

        self.out_mp4 = QtWidgets.QLineEdit("")
        self.btn_out = QtWidgets.QPushButton("Choose MP4…")
        self.btn_out.clicked.connect(self.on_choose_mp4)
        row_out = QtWidgets.QHBoxLayout()
        row_out.setSpacing(10)
        row_out.addWidget(self.out_mp4, 1)
        row_out.addWidget(self.btn_out, 0)
        io.addRow("MP4:", row_out)

        self.chk_gif = QtWidgets.QCheckBox("Also export GIF")
        self.chk_gif.setChecked(True)
        io.addRow(self.chk_gif)

        self.chk_render_clean = QtWidgets.QCheckBox("Render clean (disable edges for export)")
        self.chk_render_clean.setChecked(True)
        io.addRow(self.chk_render_clean)

        # ----------------------------
        # Camera
        # ----------------------------
        grp_cam = QtWidgets.QGroupBox("Camera (Preview + Render)")
        left_layout.addWidget(grp_cam)
        cam = QtWidgets.QFormLayout(grp_cam)
        cam.setHorizontalSpacing(10)
        cam.setVerticalSpacing(10)

        self.spin_az = self._spin(-3600, 3600, 0.5, 45.0)
        self.spin_el = self._spin(-3600, 3600, 0.5, 25.0)
        self.spin_dist = self._spin(0.01, 1e6, 0.1, 6.0)
        self.spin_cx = self._spin(-1e6, 1e6, 0.01, 0.0)
        self.spin_cy = self._spin(-1e6, 1e6, 0.01, 0.0)
        self.spin_cz = self._spin(-1e6, 1e6, 0.01, 0.0)

        for w in (self.spin_az, self.spin_el, self.spin_dist, self.spin_cx, self.spin_cy, self.spin_cz):
            w.valueChanged.connect(self.on_camera_changed)

        cam.addRow("Azimuth (deg):", self.spin_az)
        cam.addRow("Elevation (deg):", self.spin_el)
        cam.addRow("Distance:", self.spin_dist)

        center_row = QtWidgets.QHBoxLayout()
        center_row.setSpacing(10)
        center_row.addWidget(self.spin_cx)
        center_row.addWidget(self.spin_cy)
        center_row.addWidget(self.spin_cz)
        cam.addRow("Center (x,y,z):", center_row)

        self.btn_fit = QtWidgets.QPushButton("Fit camera to model")
        self.btn_fit.clicked.connect(self.fit_camera_to_model)
        cam.addRow(self.btn_fit)

        # ----------------------------
        # Materials / Colours
        # ----------------------------
        grp_mat = QtWidgets.QGroupBox("Materials / Colours")
        left_layout.addWidget(grp_mat)
        mf = QtWidgets.QFormLayout(grp_mat)
        mf.setHorizontalSpacing(10)
        mf.setVerticalSpacing(10)

        self.btn_model_color = QtWidgets.QPushButton("Pick model face colour…")
        self.btn_model_color.clicked.connect(self.pick_model_color)
        self._apply_color_button_style(self.btn_model_color, self.model_color_rgb)

        self.btn_light_color = QtWidgets.QPushButton("Pick light colour…")
        self.btn_light_color.clicked.connect(self.pick_light_color)
        self._apply_color_button_style(self.btn_light_color, self.light_color_rgb)

        mf.addRow(self.btn_model_color)
        mf.addRow(self.btn_light_color)

        # ----------------------------
        # Lighting
        # ----------------------------
        grp_light = QtWidgets.QGroupBox("Lighting")
        left_layout.addWidget(grp_light)
        lf = QtWidgets.QFormLayout(grp_light)
        lf.setHorizontalSpacing(10)
        lf.setVerticalSpacing(10)

        self.combo_light_mode = QtWidgets.QComboBox()
        self.combo_light_mode.addItems(["Headlight (camera)", "Fixed (world)"])
        self.combo_light_mode.currentTextChanged.connect(self.on_light_changed)
        lf.addRow("Mode:", self.combo_light_mode)

        self.spin_light_az = self._spin(-3600, 3600, 1.0, self.light_az_deg)
        self.spin_light_el = self._spin(-3600, 3600, 1.0, self.light_el_deg)
        self.spin_light_dist = self._spin(0.1, 1e6, 0.1, self.light_dist)

        self.spin_light_az.valueChanged.connect(self.on_light_changed)
        self.spin_light_el.valueChanged.connect(self.on_light_changed)
        self.spin_light_dist.valueChanged.connect(self.on_light_changed)

        lf.addRow("Azimuth (deg):", self.spin_light_az)
        lf.addRow("Elevation (deg):", self.spin_light_el)
        lf.addRow("Distance:", self.spin_light_dist)

        self.btn_light_to_camera = QtWidgets.QPushButton("Copy camera → light (starting point)")
        self.btn_light_to_camera.clicked.connect(self.copy_camera_to_light)
        lf.addRow(self.btn_light_to_camera)

        # NEW: brightness + diffuse sliders
        self.sld_brightness = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_brightness.setRange(0, 300)  # 0.00 .. 3.00
        self.sld_brightness.setValue(int(self.light_brightness * 100))
        self.sld_brightness.valueChanged.connect(self.on_light_params_changed)
        self.lbl_brightness = QtWidgets.QLabel(f"Brightness: {self.light_brightness:.2f}")
        lf.addRow(self.lbl_brightness, self.sld_brightness)

        self.sld_diffuse = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_diffuse.setRange(0, 200)  # 0.00 .. 2.00
        self.sld_diffuse.setValue(int(self.light_diffuse * 100))
        self.sld_diffuse.valueChanged.connect(self.on_light_params_changed)
        self.lbl_diffuse = QtWidgets.QLabel(f"Diffuse: {self.light_diffuse:.2f}")
        lf.addRow(self.lbl_diffuse, self.sld_diffuse)

        # ----------------------------
        # Display / Shading
        # ----------------------------
        grp_disp = QtWidgets.QGroupBox("Display / Shading")
        left_layout.addWidget(grp_disp)
        disp = QtWidgets.QFormLayout(grp_disp)
        disp.setVerticalSpacing(8)

        self.chk_show_faces = QtWidgets.QCheckBox("Show faces")
        self.chk_show_faces.setChecked(True)
        self.chk_show_faces.stateChanged.connect(self.apply_display_settings)

        self.chk_show_edges = QtWidgets.QCheckBox("Show edges")
        self.chk_show_edges.setChecked(True)
        self.chk_show_edges.stateChanged.connect(self.apply_display_settings)

        self.chk_wireframe_only = QtWidgets.QCheckBox("Wireframe-only (edges on, faces off)")
        self.chk_wireframe_only.setChecked(False)
        self.chk_wireframe_only.stateChanged.connect(self.on_wireframe_only_changed)

        self.chk_smooth = QtWidgets.QCheckBox("Smooth shading")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.stateChanged.connect(self.on_smooth_changed)

        self.chk_studio = QtWidgets.QCheckBox("Two-tone studio preset")
        self.chk_studio.setChecked(False)
        self.chk_studio.stateChanged.connect(self.apply_lighting_preset)

        disp.addRow(self.chk_show_faces)
        disp.addRow(self.chk_show_edges)
        disp.addRow(self.chk_wireframe_only)
        disp.addRow(self.chk_smooth)
        disp.addRow(self.chk_studio)

        self.sld_edge_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_edge_alpha.setRange(0, 100)
        self.sld_edge_alpha.setValue(85)
        self.sld_edge_alpha.valueChanged.connect(self.apply_display_settings)

        self.lbl_edge_alpha = QtWidgets.QLabel("Edge alpha: 0.85")
        disp.addRow(self.lbl_edge_alpha, self.sld_edge_alpha)

        # ----------------------------
        # Turntable Render
        # ----------------------------
        grp_render = QtWidgets.QGroupBox("Turntable Render")
        left_layout.addWidget(grp_render)
        rnd = QtWidgets.QFormLayout(grp_render)
        rnd.setHorizontalSpacing(10)
        rnd.setVerticalSpacing(10)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        rnd.addRow("Rotate axis:", self.axis_combo)

        self.spin_deg = self._spin(1, 36000, 1, 360.0)
        rnd.addRow("Total degrees:", self.spin_deg)

        self.spin_seconds = self._spin(0.1, 36000, 0.1, 6.0)
        rnd.addRow("Seconds:", self.spin_seconds)

        self.spin_fps = QtWidgets.QSpinBox()
        self.spin_fps.setRange(1, 240)
        self.spin_fps.setValue(30)
        rnd.addRow("FPS:", self.spin_fps)

        self.btn_preview = QtWidgets.QPushButton("Preview 1 frame (sanity)")
        self.btn_preview.clicked.connect(self.preview_one_frame)
        rnd.addRow(self.btn_preview)

        self.btn_render = QtWidgets.QPushButton("Render animation")
        self.btn_render.clicked.connect(self.render_animation)
        rnd.addRow(self.btn_render)

        left_layout.addSpacing(8)
        left_layout.addStretch(1)

        # Initial state
        self.apply_camera()
        self.apply_view_toggles()
        self.apply_lighting_preset()

    # ----------------------------
    # UI helpers
    # ----------------------------

    def _spin(self, mn: float, mx: float, step: float, val: float) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(mn, mx)
        s.setSingleStep(step)
        s.setDecimals(4)
        s.setValue(val)
        return s

    def _apply_color_button_style(self, btn: QtWidgets.QPushButton, rgb: Tuple[float, float, float]) -> None:
        r = int(np.clip(rgb[0], 0.0, 1.0) * 255)
        g = int(np.clip(rgb[1], 0.0, 1.0) * 255)
        b = int(np.clip(rgb[2], 0.0, 1.0) * 255)
        pix = QtGui.QPixmap(18, 18)
        pix.fill(QtGui.QColor(r, g, b))
        btn.setIcon(QtGui.QIcon(pix))
        btn.setIconSize(QtCore.QSize(18, 18))

    def _rgb255_to_float(self, rgb255: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return (rgb255[0] / 255.0, rgb255[1] / 255.0, rgb255[2] / 255.0)

    def _float_to_rgb255(self, rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (
            int(np.clip(rgb[0], 0.0, 1.0) * 255),
            int(np.clip(rgb[1], 0.0, 1.0) * 255),
            int(np.clip(rgb[2], 0.0, 1.0) * 255),
        )

    # ----------------------------
    # View toggles + background
    # ----------------------------

    def apply_view_toggles(self) -> None:
        show_grid = self.chk_show_grid.isChecked()
        show_axis = self.chk_show_axis.isChecked()

        if show_grid:
            if self.grid not in self.view.items:
                self.view.addItem(self.grid)
        else:
            if self.grid in self.view.items:
                self.view.removeItem(self.grid)

        if show_axis:
            if self.axis_item not in self.view.items:
                self.view.addItem(self.axis_item)
        else:
            if self.axis_item in self.view.items:
                self.view.removeItem(self.axis_item)

        self.view.update()

    def _apply_background(self) -> None:
        self.view.setBackgroundColor(self.bg_rgb_255)
        self.view.update()

    def pick_background_color(self) -> None:
        init = QtGui.QColor(*self.bg_rgb_255)
        col = QtWidgets.QColorDialog.getColor(init, self, "Pick background colour")
        if not col.isValid():
            return
        self.bg_rgb_255 = (col.red(), col.green(), col.blue())
        self._apply_color_button_style(self.btn_bg_color, self._rgb255_to_float(self.bg_rgb_255))
        self._apply_background()

    # ----------------------------
    # Camera + light positions
    # ----------------------------

    def apply_camera(self) -> None:
        if self._ui_busy:
            return
        self.view.opts["azimuth"] = float(self.spin_az.value())
        self.view.opts["elevation"] = float(self.spin_el.value())
        self.view.opts["distance"] = float(self.spin_dist.value())
        self.view.opts["center"] = QtGui.QVector3D(
            float(self.spin_cx.value()),
            float(self.spin_cy.value()),
            float(self.spin_cz.value()),
        )
        self.view.update()

    def camera_world_position(self) -> np.ndarray:
        center = self.view.opts.get("center", QtGui.QVector3D(0, 0, 0))
        az = math.radians(float(self.spin_az.value()))
        el = math.radians(float(self.spin_el.value()))
        dist = float(self.spin_dist.value())

        dx = math.cos(el) * math.cos(az)
        dy = math.cos(el) * math.sin(az)
        dz = math.sin(el)

        return np.array([center.x() + dx * dist, center.y() + dy * dist, center.z() + dz * dist], dtype=np.float32)

    def light_world_position(self) -> np.ndarray:
        if self.light_mode == "Headlight (camera)":
            return self.camera_world_position()

        center = self.view.opts.get("center", QtGui.QVector3D(0, 0, 0))
        az = math.radians(self.light_az_deg)
        el = math.radians(self.light_el_deg)
        dist = self.light_dist

        dx = math.cos(el) * math.cos(az)
        dy = math.cos(el) * math.sin(az)
        dz = math.sin(el)

        return np.array([center.x() + dx * dist, center.y() + dy * dist, center.z() + dz * dist], dtype=np.float32)

    # ----------------------------
    # Colour pickers
    # ----------------------------

    def pick_model_color(self) -> None:
        init = QtGui.QColor(
            int(self.model_color_rgb[0] * 255),
            int(self.model_color_rgb[1] * 255),
            int(self.model_color_rgb[2] * 255),
        )
        col = QtWidgets.QColorDialog.getColor(init, self, "Pick model face colour")
        if not col.isValid():
            return
        self.model_color_rgb = (col.redF(), col.greenF(), col.blueF())
        self._apply_color_button_style(self.btn_model_color, self.model_color_rgb)
        self.update_mesh_colors()

    def pick_light_color(self) -> None:
        init = QtGui.QColor(
            int(self.light_color_rgb[0] * 255),
            int(self.light_color_rgb[1] * 255),
            int(self.light_color_rgb[2] * 255),
        )
        col = QtWidgets.QColorDialog.getColor(init, self, "Pick light colour")
        if not col.isValid():
            return
        self.light_color_rgb = (col.redF(), col.greenF(), col.blueF())
        self._apply_color_button_style(self.btn_light_color, self.light_color_rgb)
        self.update_mesh_colors()

    # ----------------------------
    # UI events
    # ----------------------------

    def on_camera_changed(self) -> None:
        self.apply_camera()
        self.update_mesh_colors()

    def on_light_changed(self) -> None:
        if self._ui_busy:
            return
        self.light_mode = self.combo_light_mode.currentText()  # type: ignore
        self.light_az_deg = float(self.spin_light_az.value())
        self.light_el_deg = float(self.spin_light_el.value())
        self.light_dist = float(self.spin_light_dist.value())
        self.update_mesh_colors()

    def on_light_params_changed(self) -> None:
        """Brightness + diffuse sliders."""
        self.light_brightness = float(self.sld_brightness.value()) / 100.0
        self.light_diffuse = float(self.sld_diffuse.value()) / 100.0
        self.lbl_brightness.setText(f"Brightness: {self.light_brightness:.2f}")
        self.lbl_diffuse.setText(f"Diffuse: {self.light_diffuse:.2f}")
        self.update_mesh_colors()

    def copy_camera_to_light(self) -> None:
        self._ui_busy = True
        try:
            self.spin_light_az.setValue(self.spin_az.value())
            self.spin_light_el.setValue(self.spin_el.value())
            self.spin_light_dist.setValue(max(self.spin_dist.value() * 1.3, 3.0))
        finally:
            self._ui_busy = False
        self.on_light_changed()

    def on_smooth_changed(self) -> None:
        if self._verts_norm is None or self._faces_norm is None:
            return
        self.build_display_mesh()
        self.rebuild_mesh_item()
        self.update_mesh_colors()
        self.apply_display_settings()

    def apply_lighting_preset(self) -> None:
        """Studio preset: changes background only if you haven't chosen a custom background."""
        if self._ui_busy:
            return

        # If studio is toggled, we do NOT forcibly overwrite user's chosen background.
        # However, if you want "studio background", you can pick it with the background picker.
        self.update_mesh_colors()
        self.apply_display_settings()

    # ----------------------------
    # Display toggles
    # ----------------------------

    def on_wireframe_only_changed(self) -> None:
        if self._ui_busy:
            return
        if self.chk_wireframe_only.isChecked():
            self._ui_busy = True
            try:
                self.chk_show_edges.setChecked(True)
                self.chk_show_faces.setChecked(False)
            finally:
                self._ui_busy = False
        self.apply_display_settings()

    def apply_display_settings(self) -> None:
        if self._ui_busy or self.mesh_item is None:
            return

        show_faces = self.chk_show_faces.isChecked()
        show_edges = self.chk_show_edges.isChecked()

        alpha = float(self.sld_edge_alpha.value()) / 100.0
        self.lbl_edge_alpha.setText(f"Edge alpha: {alpha:.2f}")

        studio = self.chk_studio.isChecked()
        edge_rgb = self._edge_studio_rgb if studio else self._edge_default_rgb
        edge_rgba = (edge_rgb[0], edge_rgb[1], edge_rgb[2], alpha)

        _mesh_set_opt(self.mesh_item, "drawFaces", bool(show_faces))
        _mesh_set_opt(self.mesh_item, "drawEdges", bool(show_edges))
        _mesh_set_edge_color(self.mesh_item, edge_rgba)

        self.view.update()

    # ----------------------------
    # Loading
    # ----------------------------

    def on_load_obj(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select OBJ", "", "OBJ Files (*.obj);;All Files (*.*)")
        if not path:
            return
        try:
            self.load_obj(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def load_obj(self, obj_path: str) -> None:
        obj_path = resolve_obj_path(obj_path)
        mesh = load_trimesh_baked(obj_path)
        verts, faces = normalize_mesh(mesh, target_diag=2.0)

        self._verts_norm = verts
        self._faces_norm = faces
        self.bounds_radius = compute_radius(verts)

        self.lbl_obj.setText(obj_path)

        default_mp4 = os.path.join(os.path.dirname(obj_path), "turntable.mp4")
        if not self.out_mp4.text().strip():
            self.out_mp4.setText(default_mp4)

        self.build_display_mesh()
        self.rebuild_mesh_item()
        self.fit_camera_to_model()
        self.update_mesh_colors()
        self.apply_display_settings()

    def build_display_mesh(self) -> None:
        if self._verts_norm is None or self._faces_norm is None:
            return

        if self.chk_smooth.isChecked():
            v = self._verts_norm.astype(np.float32)
            f = self._faces_norm.astype(np.int32)
            n = compute_vertex_normals_smooth(v, f)
        else:
            v, f, n = make_flat_shaded_mesh(self._verts_norm, self._faces_norm)

        self._base_verts = v
        self._base_faces = f
        self._base_normals = n
        self._cur_verts = v.copy()

    # ----------------------------
    # Mesh building & updates
    # ----------------------------

    def rebuild_mesh_item(self) -> None:
        if self._cur_verts is None or self._base_faces is None:
            return

        show_faces = self.chk_show_faces.isChecked()
        show_edges = self.chk_show_edges.isChecked()

        alpha = float(self.sld_edge_alpha.value()) / 100.0
        studio = self.chk_studio.isChecked()
        edge_rgb = self._edge_studio_rgb if studio else self._edge_default_rgb
        edge_rgba = (edge_rgb[0], edge_rgb[1], edge_rgb[2], alpha)

        colors = np.ones((self._cur_verts.shape[0], 4), dtype=np.float32)
        meshdata = gl.MeshData(vertexes=self._cur_verts, faces=self._base_faces, vertexColors=colors)

        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None

        self._shader_name = choose_shader()

        self.mesh_item = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=False,
            drawFaces=show_faces,
            drawEdges=show_edges,
            edgeColor=edge_rgba,
            shader=self._shader_name,
        )
        self.view.addItem(self.mesh_item)

    def _current_lighting_params(self) -> LightingParams:
        if self.chk_studio.isChecked():
            p = self._studio_params
        else:
            p = self.lighting_params

        return LightingParams(
            base_rgb=self.model_color_rgb,
            light_rgb=self.light_color_rgb,
            ambient=p.ambient,
            diffuse=self.light_diffuse,      # NEW: user controlled
            specular=p.specular,
            shininess=p.shininess,
            brightness=self.light_brightness,  # NEW: user controlled
        )

    def update_mesh_colors(self) -> None:
        if self.mesh_item is None:
            return
        if self._cur_verts is None or self._base_faces is None or self._base_normals is None:
            return

        params = self._current_lighting_params()
        light_pos = self.light_world_position()
        cam_pos = self.camera_world_position()

        colors = bake_vertex_colors(
            verts_world=self._cur_verts,
            normals_world=self._base_normals,
            light_pos_world=light_pos,
            cam_pos_world=cam_pos,
            params=params,
        )

        meshdata = gl.MeshData(vertexes=self._cur_verts, faces=self._base_faces, vertexColors=colors)
        self.mesh_item.setMeshData(meshdata=meshdata)
        self.view.update()

    # ----------------------------
    # Camera fit
    # ----------------------------

    def fit_camera_to_model(self) -> None:
        if self._base_verts is None:
            return
        self._ui_busy = True
        try:
            self.spin_cx.setValue(0.0)
            self.spin_cy.setValue(0.0)
            self.spin_cz.setValue(0.0)
            dist = max(3.0 * self.bounds_radius, 2.5)
            self.spin_dist.setValue(dist)
        finally:
            self._ui_busy = False
        self.on_camera_changed()

    # ----------------------------
    # Output selection
    # ----------------------------

    def on_choose_mp4(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MP4", self.out_mp4.text().strip() or "", "MP4 Video (*.mp4)"
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.out_mp4.setText(path)

    # ----------------------------
    # Frame capture
    # ----------------------------

    def capture_view_rgb(self) -> np.ndarray:
        qimg = self.view.grabFramebuffer()
        qimg.setDevicePixelRatio(1.0)
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)

        w = qimg.width()
        h = qimg.height()
        bpl = qimg.bytesPerLine()

        ptr = qimg.bits()
        ptr.setsize(h * bpl)

        buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        buf = buf[:, : w * 3]
        rgb = buf.reshape((h, w, 3)).copy()
        return rgb

    def numpy_rgb_to_qimage(self, rgb: np.ndarray) -> QtGui.QImage:
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        return qimg.copy()

    # ----------------------------
    # Preview
    # ----------------------------

    def preview_one_frame(self) -> None:
        if self.mesh_item is None:
            QtWidgets.QMessageBox.warning(self, "No model", "Load an OBJ first.")
            return

        self.view.update()
        QtWidgets.QApplication.processEvents()

        img = self.capture_view_rgb()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Captured frame preview")
        layout = QtWidgets.QVBoxLayout(dlg)

        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl)

        qimg = self.numpy_rgb_to_qimage(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        lbl.setPixmap(pix.scaled(980, 560, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        dlg.resize(1020, 680)
        dlg.exec_()

    # ----------------------------
    # Render
    # ----------------------------

    def render_animation(self) -> None:
        if self.mesh_item is None or self._base_verts is None or self._base_faces is None or self._base_normals is None:
            QtWidgets.QMessageBox.warning(self, "No model", "Load an OBJ first.")
            return

        out_mp4 = self.out_mp4.text().strip()
        if not out_mp4:
            QtWidgets.QMessageBox.warning(self, "No output", "Choose an output MP4 path.")
            return
        if not out_mp4.lower().endswith(".mp4"):
            out_mp4 += ".mp4"
        out_mp4 = os.path.abspath(out_mp4)

        out_dir = os.path.dirname(out_mp4)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        axis: Axis = self.axis_combo.currentText()  # type: ignore
        degrees = float(self.spin_deg.value())
        seconds = float(self.spin_seconds.value())
        fps = int(self.spin_fps.value())
        total_frames = max(1, int(round(seconds * fps)))

        out_gif = None
        if self.chk_gif.isChecked():
            base, _ = os.path.splitext(out_mp4)
            out_gif = base + ".gif"

        settings = RenderSettings(
            fps=fps,
            seconds=seconds,
            degrees=degrees,
            axis=axis,
            out_mp4=out_mp4,
            out_gif=out_gif,
        )

        saved_faces = self.chk_show_faces.isChecked()
        saved_edges = self.chk_show_edges.isChecked()
        saved_wire = self.chk_wireframe_only.isChecked()

        if self.chk_render_clean.isChecked():
            self._ui_busy = True
            try:
                self.chk_wireframe_only.setChecked(False)
                self.chk_show_edges.setChecked(False)
                self.chk_show_faces.setChecked(True)
            finally:
                self._ui_busy = False
            self.apply_display_settings()

        prog = QtWidgets.QProgressDialog("Rendering frames…", "Cancel", 0, total_frames, self)
        prog.setWindowTitle("Rendering")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()

        writer = imageio.get_writer(
            settings.out_mp4,
            fps=settings.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

        gif_frames = []
        V0 = self._base_verts.astype(np.float32)

        self.view.update()
        QtWidgets.QApplication.processEvents()
        time.sleep(0.03)

        try:
            for i in range(total_frames):
                if prog.wasCanceled():
                    raise RuntimeError("Render cancelled by user.")

                t = i / total_frames
                ang = math.radians(t * settings.degrees)
                R = rotation_matrix(settings.axis, ang)

                self._cur_verts = (V0 @ R.T).astype(np.float32)

                self.update_mesh_colors()

                self.view.update()
                QtWidgets.QApplication.processEvents()

                frame = self.capture_view_rgb()
                writer.append_data(frame)

                if settings.out_gif:
                    gif_frames.append(frame)

                prog.setValue(i + 1)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Render failed", str(e))
            return

        finally:
            writer.close()
            prog.close()

            if self.chk_render_clean.isChecked():
                self._ui_busy = True
                try:
                    self.chk_show_faces.setChecked(saved_faces)
                    self.chk_show_edges.setChecked(saved_edges)
                    self.chk_wireframe_only.setChecked(saved_wire)
                finally:
                    self._ui_busy = False
                self.on_wireframe_only_changed()
                self.apply_display_settings()

        if settings.out_gif and gif_frames:
            imageio.mimsave(
                settings.out_gif,
                gif_frames,
                fps=settings.fps,
                loop=0,
            )

            # imageio.mimsave(settings.out_gif, gif_frames, fps=settings.fps)

        QtWidgets.QMessageBox.information(
            self,
            "Done",
            "Render complete.\n"
            f"MP4: {settings.out_mp4}\n"
            + (f"GIF: {settings.out_gif}\n" if settings.out_gif else "")
            + (f"\nShader used: {self._shader_name}" if self._shader_name else "")
        )


def main() -> None:
    pg.setConfigOptions(antialias=True)

    app = QtWidgets.QApplication([])
    app.setApplicationName("Ash's OBJ Viewer")
    apply_qt_theme(app)

    w = TurntableWindow()
    w.resize(1560, 900)
    w.show()

    app.exec_()


if __name__ == "__main__":
    main()
