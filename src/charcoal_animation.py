import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CHARCOAL_THEME = {
    "figure": "#111318",
    "axes": "#181B22",
    "text": "#E8EAED",
    "grid": "#3A3F4B",
    "trajectory": "#FF4D4D",
    "initial": "#E8EAED",
    "target": "#6EE7B7",
    "edge": "#111318",
    "cmap": "viridis",
}


def create_sphere_animation_figure(title):
    fig = plt.figure(figsize=(7, 8), facecolor=CHARCOAL_THEME["figure"])
    grid = fig.add_gridspec(2, 1, height_ratios=(5.0, 1.0), hspace=0.12)
    ax_sphere = fig.add_subplot(grid[0, 0], projection="3d")
    ax_control = fig.add_subplot(grid[1, 0])

    _style_sphere_axis(ax_sphere, title)
    _draw_unit_sphere(ax_sphere)
    _style_control_axis(ax_control)
    return fig, ax_sphere, ax_control


def add_control_state_artists(ax, gains):
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 2:
        raise ValueError("gains must have shape (frame_count, channel_count)")

    _style_control_axis(ax, gains.shape[1])
    artists = _add_directional_state_artists(ax, gains.shape[1])
    update_control_state_artists(artists, gains, 0)
    return artists


def update_control_state_artists(artists, gains, frame_index):
    values = np.asarray(gains[frame_index], dtype=float)
    states = [_control_state(value) for value in values]
    for value, state, symbol, label in zip(
        values, states, artists["symbols"], artists["labels"], strict=True
    ):
        name, color, math_symbol = _control_state_visual(state)
        symbol.set_text(math_symbol)
        symbol.set_color(color)
        label.set_text(f"{name}  {value:+.0f}")

    return tuple(artists["all"])


def _add_directional_state_artists(ax, channel_count):
    gain_labels = []
    symbols = []
    labels = []
    for index, x in enumerate(np.arange(channel_count, dtype=float) + 0.5):
        gain_labels.append(
            ax.text(
                x,
                0.82,
                f"Gain {index + 1}",
                color=CHARCOAL_THEME["text"],
                ha="center",
                va="center",
                fontsize=13,
            )
        )
        symbols.append(ax.text(x, 0.48, "", ha="center", va="center", fontsize=22))
        labels.append(
            ax.text(
                x,
                0.16,
                "",
                color=CHARCOAL_THEME["text"],
                ha="center",
                va="center",
                fontsize=9,
            )
        )
    return {
        "gain_labels": gain_labels,
        "symbols": symbols,
        "labels": labels,
        "all": [*gain_labels, *symbols, *labels],
    }


def _control_state(value):
    if np.isclose(value, 0.0):
        return "zero"
    return "normal" if value > 0.0 else "reversed"


def _control_state_visual(state):
    if state == "normal":
        return "NORMAL", CHARCOAL_THEME["target"], r"$\rightarrow$"
    if state == "reversed":
        return "REVERSED", CHARCOAL_THEME["trajectory"], r"$\leftarrow$"
    return "ZERO", CHARCOAL_THEME["initial"], r"$\times$"


def create_drone_artists_3d(
    ax,
    scale=0.2,
    body_color="#38BDF8",
    rotor_color=None,
    nose_color=None,
    alpha=1.0,
):
    rotor_color = rotor_color or CHARCOAL_THEME["initial"]
    nose_color = nose_color or CHARCOAL_THEME["trajectory"]
    arm_half = 0.82 * scale
    rotor_radius = 0.28 * scale
    rotor_centers = np.array(
        [
            [-arm_half, -arm_half, 0.0],
            [-arm_half, arm_half, 0.0],
            [arm_half, -arm_half, 0.0],
            [arm_half, arm_half, 0.0],
        ]
    )
    arm_geometry = [
        np.vstack((rotor_centers[0], rotor_centers[3])),
        np.vstack((rotor_centers[1], rotor_centers[2])),
    ]
    arm_artists = [
        ax.plot(
            [],
            [],
            [],
            color=CHARCOAL_THEME["grid"],
            linewidth=3.0,
            solid_capstyle="round",
            alpha=alpha,
            zorder=7,
        )[0]
        for _ in arm_geometry
    ]

    rotor_angle = np.linspace(0.0, 2.0 * np.pi, 30)
    rotor_geometry = [
        center
        + np.column_stack(
            (
                rotor_radius * np.cos(rotor_angle),
                rotor_radius * np.sin(rotor_angle),
                np.zeros_like(rotor_angle),
            )
        )
        for center in rotor_centers
    ]
    rotor_artists = [
        ax.plot(
            [],
            [],
            [],
            color=rotor_color,
            linewidth=2.0,
            alpha=alpha,
            zorder=8,
        )[0]
        for _ in rotor_geometry
    ]

    body_half_length = 0.43 * scale
    body_half_width = 0.32 * scale
    body_half_height = 0.14 * scale
    body_vertices = np.array(
        [
            [x, y, z]
            for z in (-body_half_height, body_half_height)
            for y in (-body_half_width, body_half_width)
            for x in (-body_half_length, body_half_length)
        ]
    )
    body_face_indices = (
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    )
    body_geometry = [body_vertices[list(indices)] for indices in body_face_indices]
    body = Poly3DCollection(
        [],
        facecolor=body_color,
        edgecolor=CHARCOAL_THEME["edge"],
        linewidth=0.8,
        alpha=alpha,
        zorder=10,
    )
    ax.add_collection3d(body)

    nose_geometry = [
        np.array(
            [
                [0.94 * scale, 0.0, 0.18 * scale],
                [0.38 * scale, -0.28 * scale, 0.18 * scale],
                [0.38 * scale, 0.28 * scale, 0.18 * scale],
            ]
        )
    ]
    nose = Poly3DCollection(
        [],
        facecolor=nose_color,
        edgecolor=CHARCOAL_THEME["edge"],
        linewidth=0.7,
        alpha=alpha,
        zorder=10,
    )
    ax.add_collection3d(nose)

    return {
        "artists": [*arm_artists, *rotor_artists, body, nose],
        "line_artists": [*arm_artists, *rotor_artists],
        "line_geometry": [*arm_geometry, *rotor_geometry],
        "poly_artists": [body, nose],
        "poly_geometry": [body_geometry, nose_geometry],
    }


def position_drone_artists_3d(drone, center, attitude):
    center = np.asarray(center, dtype=float)
    attitude = np.asarray(attitude, dtype=float)

    def transform(points):
        return center + np.asarray(points) @ attitude.T

    for artist, geometry in zip(
        drone["line_artists"], drone["line_geometry"], strict=True
    ):
        artist.set_data_3d(*transform(geometry).T)

    for artist, geometry in zip(
        drone["poly_artists"], drone["poly_geometry"], strict=True
    ):
        artist.set_verts([transform(face) for face in geometry])

    return tuple(drone["artists"])


def _style_sphere_axis(ax, title):
    ax.set_facecolor(CHARCOAL_THEME["figure"])
    ax.patch.set_alpha(0.0)
    ax.patch.set_edgecolor("none")
    ax.set_title(title, color=CHARCOAL_THEME["text"], pad=10)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=22, azim=38)
    ax.set_axis_off()

    pane_color = mpl.colors.to_rgba(CHARCOAL_THEME["axes"], 0.0)
    edge_color = mpl.colors.to_rgba(CHARCOAL_THEME["grid"], 0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_edgecolor(edge_color)
        axis.line.set_color(edge_color)
        axis._axinfo["grid"]["color"] = edge_color
        axis._axinfo["axisline"]["color"] = edge_color


def _draw_unit_sphere(ax):
    longitude = np.linspace(0.0, 2.0 * np.pi, 36)
    latitude = np.linspace(0.0, np.pi, 19)
    x = np.outer(np.cos(longitude), np.sin(latitude))
    y = np.outer(np.sin(longitude), np.sin(latitude))
    z = np.outer(np.ones_like(longitude), np.cos(latitude))
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=3,
        cstride=3,
        color=CHARCOAL_THEME["grid"],
        linewidth=0.55,
        alpha=0.65,
    )


def _style_control_axis(ax, channel_count=3):
    ax.set_facecolor(CHARCOAL_THEME["axes"])
    ax.set_title("Control Gain", color=CHARCOAL_THEME["text"], pad=4, fontsize=13)
    ax.set_xlim(0.0, float(channel_count))
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(CHARCOAL_THEME["grid"])
