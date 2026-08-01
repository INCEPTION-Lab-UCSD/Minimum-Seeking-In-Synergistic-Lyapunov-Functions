import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from charcoal_animation import (
    TRAJECTORY_COLORS,
    add_control_state_artists,
    add_trajectory_artists,
    align_control_panel,
    compact_control_panel,
    update_control_state_artists,
    update_trajectory_artists,
)
from hybrid_solution import HybridSolution

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


class Target_Seeking:
    S = np.array([[0, 1], [-1, 0]], dtype=float)
    Q = {1, 2}

    def __init__(
        self,
        z0,
        q0,
        oscillators,
        target,
        obstacle_radius,
        gamma,
        delta,
        kappa,
        epsilon,
        t_1,
        t_2,
        chi_1,
        chi_2,
        T_0,
        theta_schedule=None,
        obstacle_center=np.array([0.0, 0.0], dtype=float),
        theta_seed=None,
        T_1=1.0,
    ):

        self.q0 = q0
        self.eta0 = self._unit(oscillators)
        self.T_1 = float(T_1)
        self.target = target
        self.obstacle_radius = obstacle_radius
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.epsilon = epsilon
        self.t_1 = t_1
        self.t_2 = t_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.T_0 = T_0
        self.obstacle_center = np.asarray(obstacle_center, dtype=float)

        self.p0 = self.diffeomorphism(z0)

        self.p_target = self.diffeomorphism(target)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule(seed=theta_seed)
        else:
            self.theta_schedule = theta_schedule

    def control_gain(self, t):

        if callable(self.theta_schedule):
            return self._input_vector(self.theta_schedule(t), "theta")

        if self.theta_schedule is not None:
            value = self.theta_schedule[0][1]
            for switch_time, candidate in self.theta_schedule:
                if t < switch_time:
                    break
                value = candidate
            return self._input_vector(value, "theta")

        return np.ones(2, dtype=float)

    def diffeomorphism(self, z):
        diff = z - self.obstacle_center
        norm = np.linalg.norm(diff)
        z_unit = diff / norm
        return np.concatenate([np.array([np.log(norm - self.obstacle_radius)]), z_unit])

    def diffeomorphism_inverse(self, p):
        p = np.asarray(p)
        if p.ndim == 1:
            radius = p[0]
            unit_vec = self._unit(p[1:])

            radius = self.obstacle_radius + np.exp(radius)
            return self.obstacle_center + radius * unit_vec

        radius = self.obstacle_radius + np.exp(p[0])
        unit_vec = p[1:] / np.linalg.norm(p[1:], axis=0)
        return self.obstacle_center[:, np.newaxis] + radius * unit_vec

    def potential_function(self, unit_vec):
        target_unit_vec = self.p_target[1:]
        return 1.0 - float(np.dot(target_unit_vec, unit_vec))

    def sphere_map(self, unit_vec, q):

        return expm((1.5 - q) * self.potential_function(unit_vec) * self.S) @ unit_vec

    def synergistic_potential_function(self, unit_vec, q):
        return self.potential_function(self.sphere_map(unit_vec, q))

    def lyapunov_function(self, p, q):
        radius = p[0]
        radius_target = self.p_target[0]
        vartheta = p[1:]

        return (
            0.5 * (radius - radius_target) ** 2
            + np.sqrt((np.exp(radius) - np.exp(radius_target)) ** 2 + 1)
            - 1
            + self.synergistic_potential_function(vartheta, q)
        )

    def dynamics(self, t, y):
        p = y[:3]
        eta = y[3:-1]
        q = y[-1]
        b = self.control_vector_fields(p)

        theta = self.control_gain(t)
        u = self._input_vector(self.control(p, q, eta), "u")

        p_dot = np.sum(b * (theta * u), axis=1)
        eta_dot = 2.0 * np.pi * self.T_1**-1 * self.epsilon**-2 * (self.S @ eta)
        q_dot = 0.0

        return np.r_[p_dot, eta_dot, q_dot]

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-8):

        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2

        # state is [p = (rho, v), eta, q]
        y = np.r_[self.p0, self.eta0, self.q0]
        solution_segments = []
        while t < t_end - 1e-12:
            y = self._apply_jump(y)

            def q_jump_event(_, event_y):
                p = event_y[:3]
                q = self._mode(event_y)
                return self.delta - self.synergy_gap(p, q)

            q_jump_event.terminal = True
            q_jump_event.direction = -1

            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(t, t_end),
                y0=y,
                method="RK45",
                rtol=rtol,
                atol=atol,
                dense_output=True,
                events=q_jump_event,
            )
            solution_segments.append(sol)

            if sol.status != 1:
                break

            t = float(sol.t[-1])
            y = self.jump_map(sol.y[:, -1])

        return HybridSolution(solution_segments)

    def control(self, p, q, eta):
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1

        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.T_1 * self.kappa)
        )

        # Drive the two planar inputs from one oscillator in quadrature.
        return gain * np.array(
            [
                np.dot(direction, eta),
                np.dot(direction, self.S @ eta),
            ]
        )

    def control_vector_fields(self, p):
        p = np.asarray(p, dtype=float)
        rho = p[0]
        vartheta = self._unit(p[1:])
        radius = self.obstacle_radius + np.exp(rho)

        b = np.empty((3, 2), dtype=float)
        b[0, :] = np.exp(-rho) * vartheta
        b[1:, :] = (np.eye(2) - np.outer(vartheta, vartheta)) / radius
        return b

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def _apply_jump(self, y):
        p = y[:3]
        q = y[-1]
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return y

    def generate_theta_schedule(
        self, initial_theta=(1.0, 1.0), values=(-1.0, 0.0, 1.0), seed=None
    ):
        t_start = float(self.t_1)
        t_end = float(self.t_2)
        chi_1 = float(self.chi_1)
        chi_2 = float(self.chi_2)
        T_0 = float(self.T_0)
        values = tuple(float(value) for value in values)
        theta = tuple(float(value) for value in initial_theta)

        min_dwell = 1.0 / chi_1
        schedule = [(t_start, theta)]
        t = t_start
        monitor = T_0
        rng = np.random.default_rng(seed)

        while t < t_end - 1e-12:
            remaining = t_end - t
            duration = min(min_dwell, remaining)

            if self._theta_has_zero(theta):
                duration = min(duration, monitor / (1.0 - chi_2))
                monitor -= (1.0 - chi_2) * duration
            else:
                monitor = min(T_0, monitor + chi_2 * duration)

            t += duration
            if t >= t_end - 1e-12:
                break

            required_duration = min(min_dwell, t_end - t)
            candidates = self._theta_candidates(
                theta, values, required_duration, monitor, chi_2
            )
            theta = tuple(candidates[rng.integers(len(candidates))])
            schedule.append((t, theta))

        return schedule

    def plot_obstacle_avoidance(self, solution):
        p = solution.y[:3]
        z = self.diffeomorphism_inverse(p)

        fig = plt.figure(figsize=(7, 8.5))
        grid = fig.add_gridspec(2, 1, height_ratios=(5.2, 1.4), hspace=0.18)
        ax = fig.add_subplot(grid[0, 0])
        ax_trajectory = fig.add_subplot(grid[1, 0])
        self._style_obstacle_axis(fig, ax, z)
        boulder = self._create_boulder_patch(label="obstacle")
        ax.add_patch(boulder)

        ax.plot(
            z[0],
            z[1],
            color=CHARCOAL_THEME["trajectory"],
            linewidth=2.0,
            label="trajectory",
        )
        ax.scatter(
            z[0, 0],
            z[1, 0],
            facecolor=CHARCOAL_THEME["initial"],
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.2,
            s=70,
            label="start",
            zorder=4,
        )
        ax.scatter(
            z[0, -1],
            z[1, -1],
            facecolor=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.2,
            s=70,
            label="end",
            zorder=4,
        )
        ax.scatter(
            self.target[0],
            self.target[1],
            facecolor=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.0,
            marker="*",
            s=120,
            label="target",
            zorder=4,
        )

        legend = ax.legend(loc="best", frameon=False)
        for text in legend.get_texts():
            text.set_color(CHARCOAL_THEME["text"])
        add_trajectory_artists(
            ax_trajectory,
            solution.t,
            z.T,
            (r"$z_1$", r"$z_2$"),
            self.target,
        )
        align_control_panel(ax, ax_trajectory)
        return fig, (ax, ax_trajectory)

    def plot_trajectories_and_control_gains(self, solution):
        t = solution.t
        z = self.diffeomorphism_inverse(solution.y[:3])
        theta = np.array([self.control_gain(t_i) for t_i in t])

        fig, (ax_z, ax_theta) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            constrained_layout=True,
            gridspec_kw={"height_ratios": (4.8, 1.15)},
        )
        self._style_time_axis(fig, ax_z)

        ax_z.plot(
            t,
            z[0],
            color=CHARCOAL_THEME["trajectory"],
            linewidth=1.8,
            label="$z_1$",
        )
        ax_z.plot(
            t,
            z[1],
            color=CHARCOAL_THEME["target"],
            linewidth=1.8,
            label="$z_2$",
        )
        for index, target in enumerate(self.target):
            ax_z.axhline(
                target,
                color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
                linewidth=1.4,
                linestyle=":",
                label=rf"${['z_1', 'z_2'][index]}^\star$",
                alpha=0.9,
            )
        ax_z.set_title("Trajectories")
        ax_z.set_ylabel("$z(t)$")
        ax_z.set_xlabel("$t$")

        add_control_state_artists(ax_theta, theta, card=True)
        align_control_panel(ax_z, ax_theta)

        legend = ax_z.legend(
            handles=[*ax_z.lines[:2], *ax_z.lines[2:]],
            labels=[r"$z_1$", r"$z_2$", r"$z_1^\star$", r"$z_2^\star$"],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.0,
            frameon=True,
            facecolor=CHARCOAL_THEME["axes"],
            edgecolor=CHARCOAL_THEME["grid"],
            framealpha=0.95,
            ncol=2,
        )
        legend.set_zorder(10)
        for text in legend.get_texts():
            text.set_color(CHARCOAL_THEME["text"])

        return fig, (ax_z, ax_theta)

    def animate_obstacle_avoidance(
        self, solution, frame_count=240, interval=40, repeat_delay=1200
    ):
        frame_count = int(frame_count)
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2")

        t_frames = np.linspace(solution.t[0], solution.t[-1], frame_count)
        z = self.diffeomorphism_inverse(solution(t_frames)[:3])
        z_full = self.diffeomorphism_inverse(solution.y[:3])
        theta = np.array([self.control_gain(t_i) for t_i in t_frames])

        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(5.0, 1.55),
            width_ratios=(4.8, 1.8),
        )
        ax = fig.add_subplot(grid[0, 0])
        ax_theta = fig.add_subplot(grid[0, 1])
        ax_trajectory = fig.add_subplot(grid[1, :])
        self._style_obstacle_axis(fig, ax, z_full)
        ax.add_patch(self._create_boulder_patch())

        ax.scatter(
            z[0, 0],
            z[1, 0],
            facecolor=CHARCOAL_THEME["initial"],
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.2,
            s=80,
            zorder=4,
        )
        ax.scatter(
            z[0, -1],
            z[1, -1],
            facecolor=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            marker="*",
            linewidth=1.2,
            s=250,
            zorder=4,
        )

        drone_scale = 0.22
        drone_artists = self._create_drone_artists(ax, scale=drone_scale)
        status = ax.text(
            0.03,
            0.97,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=CHARCOAL_THEME["text"],
            family="monospace",
            fontsize=16,
        )

        control_artists = add_control_state_artists(
            ax_theta, theta, card=True, orientation="vertical"
        )
        trajectory_artists = add_trajectory_artists(
            ax_trajectory,
            t_frames,
            z.T,
            (r"$z_1$", r"$z_2$"),
            self.target,
        )
        compact_control_panel(ax, ax_theta, theta.shape[1])

        def update(frame_idx):
            x = z[0, frame_idx]
            y = z[1, frame_idx]
            if frame_idx < z.shape[1] - 1:
                direction = z[:, frame_idx + 1] - z[:, frame_idx]
            else:
                direction = z[:, frame_idx] - z[:, frame_idx - 1]
            angle = np.arctan2(direction[1], direction[0])
            self._position_drone_artists(drone_artists, x, y, angle, drone_scale)
            status.set_text(f"t = {t_frames[frame_idx]:.2f}")
            panel_artists = update_control_state_artists(
                control_artists, theta, frame_idx
            )
            path_artists = update_trajectory_artists(
                trajectory_artists, t_frames, z.T, frame_idx
            )

            return (
                *drone_artists,
                status,
                *path_artists,
                *panel_artists,
            )

        update(0)
        animation = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(t_frames)),
            interval=interval,
            repeat_delay=repeat_delay,
            blit=True,
        )
        return fig, animation

    def _style_obstacle_axis(self, fig, ax, z):
        fig.patch.set_facecolor(CHARCOAL_THEME["figure"])
        ax.set_facecolor(CHARCOAL_THEME["axes"])
        for spine in ax.spines.values():
            spine.set_color(CHARCOAL_THEME["grid"])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.title.set_color(CHARCOAL_THEME["text"])
        # ax.grid(True, color=CHARCOAL_THEME["grid"], alpha=0.45, linewidth=0.8)

        margin = 0.75
        x_values = np.r_[z[0], self.target[0], self.obstacle_center[0]]
        y_values = np.r_[z[1], self.target[1], self.obstacle_center[1]]
        ax.set_xlim(
            min(x_values.min(), self.obstacle_center[0] - self.obstacle_radius)
            - margin,
            max(x_values.max(), self.obstacle_center[0] + self.obstacle_radius)
            + margin,
        )
        ax.set_ylim(
            min(y_values.min(), self.obstacle_center[1] - self.obstacle_radius)
            - margin,
            max(y_values.max(), self.obstacle_center[1] + self.obstacle_radius)
            + margin,
        )

        ax.set_title("Obstacle Avoidance")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal", adjustable="box")

    def _style_time_axis(self, fig, ax):
        fig.patch.set_facecolor(CHARCOAL_THEME["figure"])
        ax.set_facecolor(CHARCOAL_THEME["axes"])
        for spine in ax.spines.values():
            spine.set_color(CHARCOAL_THEME["grid"])

        ax.tick_params(colors=CHARCOAL_THEME["text"])
        ax.xaxis.label.set_color(CHARCOAL_THEME["text"])
        ax.yaxis.label.set_color(CHARCOAL_THEME["text"])
        ax.title.set_color(CHARCOAL_THEME["text"])
        ax.grid(True, color=CHARCOAL_THEME["grid"], alpha=0.45, linewidth=0.8)

    def _create_boulder_patch(self, label=None):
        angles = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        radial_jitter = np.array(
            [
                0.99,
                0.89,
                1.00,
                0.96,
                0.98,
                0.91,
                0.99,
                0.93,
                0.97,
                0.87,
                1.00,
                0.95,
                0.99,
                0.90,
                0.98,
                0.94,
                1.00,
                0.92,
            ]
        )
        radii = self.obstacle_radius * radial_jitter
        vertices = self.obstacle_center + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        return Polygon(
            vertices,
            closed=True,
            facecolor=CHARCOAL_THEME["grid"],
            edgecolor=CHARCOAL_THEME["text"],
            linewidth=1.2,
            alpha=0.9,
            label=label,
            zorder=2,
        )

    def _create_drone_artists(self, ax, scale):
        rotor_radius = 0.55 * scale
        arm_length = 1.8 * scale
        arm_width = 0.12 * scale
        body = patches.FancyBboxPatch(
            (-0.72 * scale, -0.50 * scale),
            1.44 * scale,
            1.00 * scale,
            boxstyle=f"round,pad={0.08 * scale},rounding_size={0.35 * scale}",
            facecolor="#38BDF8",
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.0,
            zorder=8,
        )
        arms = [
            ax.plot(
                [-arm_length / 2, arm_length / 2],
                [-arm_length / 2, arm_length / 2],
                color=CHARCOAL_THEME["edge"],
                linewidth=arm_width * 90,
                solid_capstyle="round",
                zorder=6,
            )[0],
            ax.plot(
                [-arm_length / 2, arm_length / 2],
                [arm_length / 2, -arm_length / 2],
                color=CHARCOAL_THEME["edge"],
                linewidth=arm_width * 90,
                solid_capstyle="round",
                zorder=6,
            )[0],
        ]

        rotors = [
            patches.Circle(
                (x, y),
                rotor_radius,
                facecolor=CHARCOAL_THEME["initial"],
                edgecolor=CHARCOAL_THEME["edge"],
                linewidth=0.8,
                alpha=0.95,
                zorder=7,
            )
            for x, y in (
                (-arm_length / 2, -arm_length / 2),
                (-arm_length / 2, arm_length / 2),
                (arm_length / 2, -arm_length / 2),
                (arm_length / 2, arm_length / 2),
            )
        ]
        rotor_hubs = [
            patches.Circle(
                rotor.center,
                rotor_radius * 0.33,
                facecolor="#111318",
                edgecolor=CHARCOAL_THEME["edge"],
                linewidth=0.6,
                zorder=9,
            )
            for rotor in rotors
        ]
        nose = patches.Polygon(
            [
                (1.10 * scale, 0.0),
                (0.48 * scale, -0.36 * scale),
                (0.48 * scale, 0.36 * scale),
            ],
            closed=True,
            facecolor="#FF4D4D",
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=0.8,
            zorder=9,
        )

        patch_artists = [*rotors, body, *rotor_hubs, nose]
        for artist in patch_artists:
            ax.add_patch(artist)

        return [*arms, *patch_artists]

    def _position_drone_artists(self, artists, x, y, angle, scale):
        transform = Affine2D().rotate(angle).translate(x, y)
        for artist in artists:
            artist.set_transform(transform + artist.axes.transData)

    @classmethod
    def _theta_candidates(cls, theta, values, required_duration, monitor, chi_2):
        candidates = [
            candidate
            for candidate in cls._theta_space(values, len(theta))
            if candidate != theta
        ]

        zero_budget_required = (1.0 - chi_2) * required_duration
        if monitor < zero_budget_required - 1e-12:
            candidates = [
                candidate
                for candidate in candidates
                if not cls._theta_has_zero(candidate)
            ]

        return candidates

    @staticmethod
    def _theta_space(values, dimension):
        if dimension == 0:
            return [()]

        tails = Target_Seeking._theta_space(values, dimension - 1)
        return [(value, *tail) for value in values for tail in tails]

    @staticmethod
    def _theta_has_zero(theta):
        return any(np.isclose(value, 0.0) for value in theta)

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float).copy()
        p = y_plus[:3]
        y_plus[-1] = self.argmin_mode(p)
        return y_plus

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values) + 1)

    def _unit(self, vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _input_vector(value, name):
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size == 1:
            return np.full(2, vector[0], dtype=float)
        if vector.size != 2:
            raise ValueError(
                f"{name} must be scalar or length 2, got shape {vector.shape}"
            )
        return vector

    def _mode(self, y):
        q = y[-1]
        return int(round(q))
