import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
    ax_gain = fig.add_subplot(grid[1, 0])

    _style_sphere_axis(ax_sphere, title)
    _draw_unit_sphere(ax_sphere)
    _style_gain_axis(ax_gain)
    return fig, ax_sphere, ax_gain


def add_control_gain_artists(ax, times, gains):
    gains = np.asarray(gains, dtype=float)
    cmap = mpl.colormaps[CHARCOAL_THEME["cmap"]]
    colors = [cmap(value) for value in (0.15, 0.55, 0.9)]
    lines = []

    for index, color in enumerate(colors):
        (line,) = ax.plot(
            [],
            [],
            color=color,
            linewidth=1.8,
            drawstyle="steps-post",
            label=rf"$\theta_{index + 1}$",
        )
        lines.append(line)

    marker = ax.scatter(
        np.full(gains.shape[1], times[0]),
        gains[0],
        c=colors,
        s=20,
        zorder=4,
    )
    ax.set_xlim(float(times[0]), float(times[-1]))
    margin = max(0.2, 0.08 * float(np.ptp(gains)))
    ax.set_ylim(float(gains.min() - margin), float(gains.max() + margin))
    legend = ax.legend(loc="upper right", ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_color(CHARCOAL_THEME["text"])
    return lines, marker


def update_control_gain_artists(lines, marker, times, gains, frame_index):
    frame_slice = slice(0, frame_index + 1)
    for index, line in enumerate(lines):
        line.set_data(times[frame_slice], gains[frame_slice, index])
    marker.set_offsets(
        np.column_stack(
            (
                np.full(gains.shape[1], times[frame_index]),
                gains[frame_index],
            )
        )
    )


def _style_sphere_axis(ax, title):
    ax.set_facecolor(CHARCOAL_THEME["axes"])
    ax.set_title(title, color=CHARCOAL_THEME["text"], pad=10)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=22, azim=38)

    pane_color = mpl.colors.to_rgba(CHARCOAL_THEME["axes"], 1.0)
    edge_color = mpl.colors.to_rgba(CHARCOAL_THEME["grid"], 0.6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_edgecolor(edge_color)


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


def _style_gain_axis(ax):
    ax.set_facecolor(CHARCOAL_THEME["axes"])
    ax.set_title("Control Gains", color=CHARCOAL_THEME["text"], pad=4)
    ax.set_xlabel("$t$", color=CHARCOAL_THEME["text"])
    ax.set_ylabel(r"$\theta(t)$", color=CHARCOAL_THEME["text"])
    ax.tick_params(colors=CHARCOAL_THEME["text"], length=3)
    ax.grid(color=CHARCOAL_THEME["grid"], alpha=0.55, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(CHARCOAL_THEME["grid"])
