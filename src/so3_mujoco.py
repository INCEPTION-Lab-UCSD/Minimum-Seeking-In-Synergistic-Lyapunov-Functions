"""MuJoCo renderer for the SO(3) minimum-seeking example.

The controller remains in :mod:`so3`; this module only replays a solved
attitude trajectory on a spherical MuJoCo rigid body and writes a video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_MODEL_XML = r"""
<mujoco model="so3_attitude_replay">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="135" elevation="-20"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.65 0.65 0.65" specular="0.25 0.25 0.25"/>
    <rgba haze="0.05 0.07 0.10 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.025 0.035 0.060" rgb2="0.16 0.20 0.28" width="512" height="3072"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.11 0.13 0.17" rgb2="0.16 0.18 0.23" width="512" height="512"/>
    <material name="floor" texture="floor_tex" texrepeat="8 8" reflectance="0.12" roughness="0.8"/>
    <material name="blue" rgba="0.10 0.62 0.98 1" metallic="0.35" roughness="0.24"/>
    <material name="target" rgba="0.34 0.92 0.67 0.18" emission="0.18" roughness="0.35"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0" density="300"/>
  </default>
  <worldbody>
    <light name="key" pos="-3 -4 7" dir="0.35 0.4 -1" directional="true" diffuse="0.85 0.88 1"/>
    <light name="fill" pos="4 1 4" dir="-0.6 -0.1 -1" directional="true" diffuse="0.35 0.42 0.55"/>
    <geom name="floor" type="plane" size="7 7 0.1" material="floor" contype="1" conaffinity="1"/>
    <geom name="pad_outer" type="cylinder" pos="0 0 0.012" size="1.22 0.012" rgba="0.12 0.15 0.20 1"/>
    <geom name="pad_inner" type="cylinder" pos="0 0 0.026" size="1.04 0.014" rgba="0.07 0.09 0.13 1"/>
    <geom type="capsule" fromto="-1.55 0 0.05 1.55 0 0.05" size="0.012" rgba="0.45 0.16 0.18 0.65"/>
    <geom type="capsule" fromto="0 -1.55 0.05 0 1.55 0.05" size="0.012" rgba="0.16 0.42 0.24 0.65"/>

    <camera name="hero" pos="4 -6 3.2"
            xyaxes="0.83205 0.55470 0 -0.14825 0.22237 0.96362"/>

    <!-- Desired attitude: a translucent, slightly larger reference ghost. -->
    <body name="target" mocap="true" pos="0 0 1.25">
      <geom type="sphere" size="0.39" material="target"/>
      <geom type="capsule" fromto="0 0 0 1.08 0 0" size="0.014" rgba="1 0.28 0.26 0.22"/>
      <geom type="capsule" fromto="0 0 0 0 1.08 0" size="0.014" rgba="0.35 1 0.48 0.22"/>
      <geom type="capsule" fromto="0 0 0 0 0 1.08" size="0.014" rgba="0.28 0.58 1 0.22"/>
    </body>

    <!-- Actual attitude, driven directly from the SO(3) solution. -->
    <body name="attitude" pos="0 0 1.25">
      <freejoint name="attitude_free"/>
      <geom name="body" type="sphere" size="0.36" material="blue"/>
      <geom name="spot_x" type="cylinder" pos="0.36 0 0"
            quat="0.707107 0 0.707107 0" size="0.085 0.008"
            rgba="1 0.20 0.18 1"/>
      <geom name="spot_y" type="cylinder" pos="0 0.36 0"
            quat="0.707107 -0.707107 0 0" size="0.085 0.008"
            rgba="0.20 0.92 0.36 1"/>
      <geom name="spot_z" type="cylinder" pos="0 0 0.36"
            size="0.085 0.008" rgba="0.20 0.48 1 1"/>
      <geom type="capsule" fromto="0 0 0 0.82 0 0" size="0.018" rgba="1 0.20 0.18 0.95"/>
      <geom type="sphere" pos="0.82 0 0" size="0.035" rgba="1 0.20 0.18 1"/>
      <geom type="capsule" fromto="0 0 0 0 0.82 0" size="0.018" rgba="0.20 0.92 0.36 0.95"/>
      <geom type="sphere" pos="0 0.82 0" size="0.035" rgba="0.20 0.92 0.36 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.82" size="0.018" rgba="0.20 0.48 1 0.95"/>
      <geom type="sphere" pos="0 0 0.82" size="0.035" rgba="0.20 0.48 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def rotation_to_mujoco_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3-by-3 rotation matrix to MuJoCo's ``w, x, y, z`` order."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(matrix).reshape(-1))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def attitude_error_degrees(rotation: np.ndarray, target: np.ndarray) -> float:
    """Return the geodesic rotation error in degrees."""
    cosine = (np.trace(np.asarray(target).T @ np.asarray(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def render_so3_animation(
    simulation,
    solution,
    output_path: str | Path = "Animations/so3_mujoco.mp4",
    *,
    fps: int = 30,
    frame_count: int | None = None,
    width: int = 960,
    height: int = 720,
) -> Path:
    """Render ``solution`` as an MP4 or GIF and return the output path."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    t_start = max(simulation.t_1, float(solution.t[0]))
    t_end = min(simulation.t_2, float(solution.t[-1]))
    if frame_count is None:
        frame_count = max(2, int(np.ceil((t_end - t_start) * fps)) + 1)
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError("output_path must end in .mp4 or .gif")

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    free_qpos = model.joint("attitude_free").qposadr[0]
    target_mocap = model.body_mocapid[model.body("target").id]
    data.mocap_pos[target_mocap] = np.array([0.0, 0.0, 1.25])
    data.mocap_quat[target_mocap] = rotation_to_mujoco_quaternion(simulation.R_target)

    times = np.linspace(t_start, t_end, frame_count)
    states = solution(times)
    gains = np.vstack([simulation._get_control_gain(time) for time in times])
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = _video_writer(output, fps)

    try:
        for index, time in enumerate(times):
            rotation = simulation._project_so3(simulation._matrix(states[:9, index]))
            data.qpos[free_qpos : free_qpos + 3] = (0.0, 0.0, 1.25)
            data.qpos[free_qpos + 3 : free_qpos + 7] = rotation_to_mujoco_quaternion(
                rotation
            )
            data.qvel[:] = 0.0
            data.time = float(time)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera="hero")
            frame = renderer.render()
            annotated = _annotate_frame(
                frame,
                time=float(time),
                t_start=t_start,
                t_end=t_end,
                mode=simulation._mode(states[:, index]),
                potential=simulation.lyapunov_function(
                    states[:9, index], simulation._mode(states[:, index])
                ),
                gap=simulation.synergy_gap(
                    states[:9, index], simulation._mode(states[:, index])
                ),
                error_degrees=attitude_error_degrees(rotation, simulation.R_target),
                gains=gains[index],
            )
            writer.append_data(annotated)
    finally:
        writer.close()
        renderer.close()

    return output


def _video_writer(output: Path, fps: int):
    if output.suffix.lower() == ".gif":
        return imageio.get_writer(output, mode="I", fps=fps, loop=0)
    return imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def _annotate_frame(
    frame: np.ndarray,
    *,
    time: float,
    t_start: float,
    t_end: float,
    mode: int,
    potential: float,
    gap: float,
    error_degrees: float,
    gains: np.ndarray,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    scale = min(width / 960.0, height / 720.0)
    margin = int(24 * scale)
    panel_width = int(520 * scale)
    panel_height = int(230 * scale)
    panel_left = (width - panel_width) // 2
    panel_top = margin
    radius = int(16 * scale)
    draw.rounded_rectangle(
        (
            panel_left,
            panel_top,
            panel_left + panel_width,
            panel_top + panel_height,
        ),
        radius=radius,
        fill=(8, 12, 20, 205),
        outline=(104, 119, 142, 125),
        width=max(1, int(1.5 * scale)),
    )

    title_font = _font(int(29 * scale), bold=True)
    body_font = _font(int(21 * scale))
    small_font = _font(int(18 * scale))
    x = panel_left + int(26 * scale)
    y = panel_top + int(15 * scale)
    panel_title = "SO(3) STABILIZATION"
    draw.text(
        (panel_left + panel_width // 2, y),
        panel_title,
        font=title_font,
        fill=(238, 243, 250, 255),
        anchor="mt",
    )
    divider_y = panel_top + int(61 * scale)
    draw.line(
        (
            panel_left + int(24 * scale),
            divider_y,
            panel_left + panel_width - int(24 * scale),
            divider_y,
        ),
        fill=(104, 119, 142, 90),
        width=max(1, int(scale)),
    )

    y = panel_top + int(76 * scale)
    draw.text(
        (x, y), f"Time = {time:5.2f} s", font=body_font, fill=(221, 228, 239, 255)
    )
    mode_text = f"Mode q = {mode}"
    mode_box = draw.textbbox((0, 0), mode_text, font=body_font)
    draw.text(
        (
            panel_left + panel_width - int(26 * scale) - (mode_box[2] - mode_box[0]),
            y,
        ),
        mode_text,
        font=body_font,
        fill=(110, 231, 183, 255),
    )

    y += int(36 * scale)
    draw.text(
        (x, y),
        f"Error {error_degrees:6.2f} deg",
        font=body_font,
        fill=(221, 228, 239, 255),
    )
    y += int(32 * scale)
    draw.text(
        (x, y),
        f"Potential = {potential:7.4f}",
        font=small_font,
        fill=(178, 190, 207, 255),
    )
    gap_text = f"Gap = {gap:7.4f}"
    gap_box = draw.textbbox((0, 0), gap_text, font=small_font)
    draw.text(
        (
            panel_left + panel_width - int(26 * scale) - (gap_box[2] - gap_box[0]),
            y,
        ),
        gap_text,
        font=small_font,
        fill=(178, 190, 207, 255),
    )
    y += int(33 * scale)
    gain_text = ",  ".join(f"Gain{i + 1} {value:+.0f}" for i, value in enumerate(gains))
    draw.text(
        (panel_left + panel_width // 2, y),
        gain_text,
        font=small_font,
        fill=(147, 197, 253, 255),
        anchor="mt",
    )

    progress = (
        1.0 if np.isclose(t_end, t_start) else (time - t_start) / (t_end - t_start)
    )
    bar_left = x
    bar_right = panel_left + panel_width - int(26 * scale)
    bar_top = panel_top + panel_height - int(18 * scale)
    bar_height = max(3, int(6 * scale))
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, bar_top + bar_height),
        radius=bar_height,
        fill=(62, 73, 91, 230),
    )
    draw.rounded_rectangle(
        (
            bar_left,
            bar_top,
            bar_left + int((bar_right - bar_left) * np.clip(progress, 0.0, 1.0)),
            bar_top + bar_height,
        ),
        radius=bar_height,
        fill=(56, 189, 248, 255),
    )
    return np.asarray(image)


def _font(size: int, *, bold: bool = False):
    candidates = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/SFNS.ttf"
        ),
        "/System/Library/Fonts/Helvetica.ttc",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(size, 10))
        except OSError:
            pass
    return ImageFont.load_default()


def _default_simulation():
    from so3 import SO3

    simulation = SO3(
        p0=np.diag([-1.0, 1.0, -1.0]),
        eta0=np.tile([1.0, 0.0], (3, 1)),
        q0=1,
        target=np.eye(3),
        gamma=1.0,
        delta=0.2,
        kappa=4.0,
        epsilon=1.0 / np.sqrt(12.0 * np.pi),
        t_1=0.0,
        t_2=10.0,
        chi_1=1.0,
        chi_2=0.5,
        control_gain_constants=np.array([1.0, 2.0, 3.0]),
        angular_velocity=np.array([11.0, 12.0, 13.0]),
        theta_seed=0,
    )
    return simulation, simulation.solve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("Animations/so3_mujoco.mp4")
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    simulation, solution = _default_simulation()
    output = render_so3_animation(
        simulation,
        solution,
        args.output,
        fps=args.fps,
        frame_count=args.frames,
        width=args.width,
        height=args.height,
    )
    print(output)


if __name__ == "__main__":
    main()
