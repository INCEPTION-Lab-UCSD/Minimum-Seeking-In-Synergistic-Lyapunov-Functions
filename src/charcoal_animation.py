import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
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

TRAJECTORY_COLORS = (
    CHARCOAL_THEME["trajectory"],
    CHARCOAL_THEME["target"],
    "#38BDF8",
    "#FBBF24",
)

CONTROL_HEADER_INCHES = 0.62
CONTROL_ROW_INCHES = 0.56
CONTROL_BOTTOM_INCHES = 0.08


def create_sphere_animation_figure(title):
    fig = plt.figure(figsize=(9.5, 8), facecolor=CHARCOAL_THEME["figure"])
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(5.4, 1.6),
        width_ratios=(4.8, 1.8),
        hspace=0.22,
        wspace=0.12,
    )
    ax_sphere = fig.add_subplot(grid[0, 0], projection="3d")
    ax_control = fig.add_subplot(grid[0, 1])
    ax_trajectory = fig.add_subplot(grid[1, :])

    _style_sphere_axis(ax_sphere, title)
    _draw_unit_sphere(ax_sphere)
    _style_trajectory_axis(ax_trajectory)
    return fig, ax_sphere, ax_trajectory, ax_control


def add_trajectory_artists(ax, times, values, labels, target_values=None):
    """Create a progressive time-trajectory plot on a dedicated axis."""
    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.shape[0] != times.size:
        raise ValueError("values must have one row per time")
    if values.shape[1] != len(labels):
        raise ValueError("labels must contain one entry per trajectory channel")

    if target_values is not None:
        target_values = np.asarray(target_values, dtype=float).reshape(-1)
        if target_values.size != values.shape[1]:
            raise ValueError(
                "target_values must contain one entry per trajectory channel"
            )

    _style_trajectory_axis(ax)
    lines = [
        ax.plot(
            [],
            [],
            color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
            linewidth=1.7,
            label=label,
        )[0]
        for index, label in enumerate(labels)
    ]
    points = [
        ax.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
            markersize=3.5,
        )[0]
        for index in range(values.shape[1])
    ]
    targets = []
    if target_values is not None:
        targets = [
            ax.axhline(
                target,
                color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
                linewidth=1.4,
                # Shift each dot pattern so coincident targets remain visible.
                linestyle=(index * 1.5, (1.0, 2.8)),
                label=rf"${labels[index].strip('$')}^\star$",
                alpha=0.9,
            )
            for index, target in enumerate(target_values)
        ]
    cursor = ax.axvline(
        times[0],
        color=CHARCOAL_THEME["initial"],
        linewidth=1.0,
        alpha=0.65,
    )

    ax.set_xlim(times[0], times[-1])
    plotted_values = (
        values
        if target_values is None
        else np.concatenate((values.ravel(), target_values))
    )
    finite_values = plotted_values[np.isfinite(plotted_values)]
    if finite_values.size:
        lower = float(finite_values.min())
        upper = float(finite_values.max())
        span = upper - lower
        margin = 0.08 * span if span > 1e-9 else max(0.1, 0.08 * abs(upper))
        ax.set_ylim(lower - margin, upper + margin)

    target_legend_handles = [
        Line2D(
            [],
            [],
            color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
            linewidth=1.4,
            linestyle=":",
        )
        for index in range(len(targets))
    ]
    legend = ax.legend(
        handles=[*lines, *target_legend_handles],
        labels=[line.get_label() for line in lines]
        + [target.get_label() for target in targets],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0.0,
        frameon=True,
        facecolor=CHARCOAL_THEME["axes"],
        edgecolor=CHARCOAL_THEME["grid"],
        framealpha=0.95,
        ncol=2,
        fontsize=8.5,
    )
    legend.set_zorder(10)
    for text in legend.get_texts():
        text.set_color(CHARCOAL_THEME["text"])

    artists = {
        "lines": lines,
        "points": points,
        "cursor": cursor,
        "targets": targets,
        "all": [*lines, *points, *targets, cursor],
    }
    update_trajectory_artists(artists, times, values, len(times) - 1)
    return artists


def update_trajectory_artists(artists, times, values, frame_index):
    """Show the trajectory history through ``frame_index``."""
    end = int(frame_index) + 1
    for channel, line in enumerate(artists["lines"]):
        line.set_data(times[:end], values[:end, channel])
        artists["points"][channel].set_data(
            [times[frame_index]], [values[frame_index, channel]]
        )
    artists["cursor"].set_xdata([times[frame_index], times[frame_index]])
    return tuple(artists["all"])


def add_control_state_artists(ax, gains, *, card=False, orientation="horizontal"):
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 2:
        raise ValueError("gains must have shape (frame_count, channel_count)")

    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    _style_control_axis(ax, gains.shape[1], card=card, orientation=orientation)
    artists = _add_directional_state_artists(
        ax, gains.shape[1], card=card, orientation=orientation
    )
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


def align_control_panel(ax_reference, ax_control):
    """Match a control panel's displayed width to the plot directly above it."""
    fig = ax_reference.figure
    if ax_control.figure is not fig:
        raise ValueError("ax_reference and ax_control must belong to the same figure")

    # Equal-aspect trajectory axes can occupy only part of their grid cell. Resolve
    # the layout once, then use that displayed width for the panel beneath it.
    fig.canvas.draw()
    reference_position = ax_reference.get_position().frozen()
    control_position = ax_control.get_position().frozen()
    fig.set_layout_engine(None)
    ax_reference.set_position(reference_position)
    ax_control.set_position(
        (
            reference_position.x0,
            control_position.y0,
            reference_position.width,
            control_position.height,
        )
    )


def compact_control_panel(ax_reference, ax_control, channel_count):
    """Size and vertically center a sidebar in the reference panel."""
    if channel_count < 1:
        raise ValueError("channel_count must be positive")

    fig = ax_reference.figure
    if ax_control.figure is not fig:
        raise ValueError("ax_reference and ax_control must belong to the same figure")

    fig.canvas.draw()
    reference_position = ax_reference.get_position().frozen()
    control_position = ax_control.get_position().frozen()
    target_height_inches = (
        CONTROL_HEADER_INCHES
        + CONTROL_ROW_INCHES * channel_count
        + CONTROL_BOTTOM_INCHES
    )
    target_height = min(
        reference_position.height,
        target_height_inches / fig.get_figheight(),
    )

    fig.set_layout_engine(None)
    ax_reference.set_position(reference_position)
    ax_control.set_position(
        (
            control_position.x0,
            reference_position.y0 + 0.5 * (reference_position.height - target_height),
            control_position.width,
            target_height,
        )
    )


def _add_directional_state_artists(
    ax, channel_count, *, card=False, orientation="horizontal"
):
    if orientation == "vertical":
        gain_labels = []
        symbols = []
        labels = []
        _, _, row_centers, _ = _vertical_control_layout(channel_count)

        for index, y in enumerate(row_centers):
            gain_labels.append(
                ax.text(
                    0.08,
                    y,
                    f"Gain {index + 1}",
                    color=CHARCOAL_THEME["text"],
                    ha="left",
                    va="center",
                    fontsize=10,
                )
            )
            symbols.append(
                ax.text(
                    0.43,
                    y,
                    "",
                    ha="center",
                    va="center",
                    fontsize=18,
                )
            )
            labels.append(
                ax.text(
                    0.58,
                    y,
                    "",
                    color=CHARCOAL_THEME["text"],
                    ha="left",
                    va="center",
                    fontsize=8,
                )
            )
        return {
            "gain_labels": gain_labels,
            "symbols": symbols,
            "labels": labels,
            "all": [*gain_labels, *symbols, *labels],
        }

    if card:
        gain_y, symbol_y, label_y = 0.56, 0.33, 0.13
    else:
        gain_y, symbol_y, label_y = 0.82, 0.48, 0.16

    gain_labels = []
    symbols = []
    labels = []
    for index, x in enumerate(np.arange(channel_count, dtype=float) + 0.5):
        gain_labels.append(
            ax.text(
                x,
                gain_y,
                f"Gain {index + 1}",
                color=CHARCOAL_THEME["text"],
                ha="center",
                va="center",
                fontsize=12 if card else 13,
            )
        )
        symbols.append(ax.text(x, symbol_y, "", ha="center", va="center", fontsize=22))
        labels.append(
            ax.text(
                x,
                label_y,
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
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.24)
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


def _style_trajectory_axis(ax):
    ax.set_facecolor(CHARCOAL_THEME["axes"])
    for spine in ax.spines.values():
        spine.set_color(CHARCOAL_THEME["grid"])
    ax.tick_params(colors=CHARCOAL_THEME["text"], labelsize=8)
    ax.xaxis.label.set_color(CHARCOAL_THEME["text"])
    ax.yaxis.label.set_color(CHARCOAL_THEME["text"])
    ax.set_title(
        "STATE TRAJECTORY",
        color=CHARCOAL_THEME["text"],
        fontsize=9.5,
        fontweight="bold",
        loc="left",
        pad=5,
    )
    ax.set_xlabel("t", fontsize=8)
    ax.grid(True, color=CHARCOAL_THEME["grid"], alpha=0.45, linewidth=0.7)


def _draw_unit_sphere(ax):
    longitude = np.linspace(0.0, 2.0 * np.pi, 36)
    latitude = np.linspace(0.0, np.pi, 19)
    x = np.outer(np.cos(longitude), np.sin(latitude))
    y = np.outer(np.sin(longitude), np.sin(latitude))
    z = np.outer(np.ones_like(longitude), np.cos(latitude))
    ax.plot_surface(
        x,
        y,
        z,
        color="#334155",
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.12,
    )
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=3,
        cstride=3,
        color=CHARCOAL_THEME["grid"],
        linewidth=0.6,
        alpha=0.72,
    )


def _style_control_axis(ax, channel_count=3, *, card=False, orientation="horizontal"):
    ax.set_facecolor(CHARCOAL_THEME["figure"] if card else CHARCOAL_THEME["axes"])
    if card:
        ax.set_title("")
        header_center, divider_y, _, separators = (
            _vertical_control_layout(channel_count)
            if orientation == "vertical"
            else (0.84, 0.72, (), ())
        )
        frame = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            transform=ax.transAxes,
            facecolor=CHARCOAL_THEME["axes"],
            edgecolor=CHARCOAL_THEME["grid"],
            linewidth=1.1,
            clip_on=False,
            zorder=-2,
        )
        ax.add_patch(frame)
        ax.text(
            0.045,
            header_center,
            "CONTROL GAIN",
            transform=ax.transAxes,
            color=CHARCOAL_THEME["text"],
            alpha=0.78,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.plot(
            [0.04, 0.96],
            [divider_y, divider_y],
            transform=ax.transAxes,
            color=CHARCOAL_THEME["grid"],
            linewidth=0.8,
            alpha=0.8,
        )
        if orientation == "vertical":
            for y in separators:
                ax.plot(
                    [0.06, 0.94],
                    [y, y],
                    color=CHARCOAL_THEME["grid"],
                    linewidth=0.8,
                    alpha=0.65,
                )
        else:
            for index in range(1, channel_count):
                ax.plot(
                    [index, index],
                    [0.10, 0.64],
                    color=CHARCOAL_THEME["grid"],
                    linewidth=0.8,
                    alpha=0.65,
                )
    else:
        ax.set_title("Control Gain", color=CHARCOAL_THEME["text"], pad=-2, fontsize=13)
    ax.set_xlim(0.0, 1.0 if orientation == "vertical" else float(channel_count))
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(not card)
        spine.set_color(CHARCOAL_THEME["grid"])


def _vertical_control_layout(channel_count):
    total_inches = (
        CONTROL_HEADER_INCHES
        + CONTROL_ROW_INCHES * channel_count
        + CONTROL_BOTTOM_INCHES
    )
    header_fraction = CONTROL_HEADER_INCHES / total_inches
    row_fraction = CONTROL_ROW_INCHES / total_inches
    header_center = 1.0 - 0.5 * header_fraction
    divider_y = 1.0 - header_fraction
    row_centers = np.array(
        [divider_y - (index + 0.5) * row_fraction for index in range(channel_count)]
    )
    separators = np.array(
        [divider_y - index * row_fraction for index in range(1, channel_count)]
    )
    return header_center, divider_y, row_centers, separators
