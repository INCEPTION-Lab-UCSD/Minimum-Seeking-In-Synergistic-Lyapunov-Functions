"""Hybrid minimum-seeking control for a planar nonholonomic vehicle.

The module name keeps the repository's original ``nonholomonic`` spelling for
backwards compatibility.  The model implemented here is the nonholonomic
system from equations (35)--(38) of the accompanying paper.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from charcoal_animation import (
    CHARCOAL_THEME,
    add_control_state_artists,
    update_control_state_artists,
)
from hybrid_solution import HybridSolution


class Nonholonomic:
    """Simulate hybrid target seeking with nonholonomic kinematics.

    The continuous state is ordered as ``[z, psi, eta, q]``, where ``z`` is
    the physical position, ``psi`` is the vehicle heading, ``eta`` is the
    extremum-seeking oscillator, and ``q`` is the hybrid potential mode.
    """

    Q = (1, 2)
    S = np.array([[0.0, 1.0], [-1.0, 0.0]])
    e1 = np.array([1.0, 0.0])

    def __init__(
        self,
        z0,
        psi0,
        eta0,
        q0,
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
        T_0=1.0,
        T_1=1.0,
        theta_schedule=None,
        obstacle_center=(0.0, 0.0),
        theta_seed=None,
    ):
        self.obstacle_center = self._vector(obstacle_center, "obstacle_center")
        self.obstacle_radius = float(obstacle_radius)
        if self.obstacle_radius <= 0.0:
            raise ValueError("obstacle_radius must be positive")

        self.z0 = self._outside_obstacle(z0, "z0")
        self.target = self._outside_obstacle(target, "target")
        self.psi0 = self._unit(psi0, "psi0")
        self.eta0 = self._unit(eta0, "eta0")
        self.q0 = self._validate_mode(q0)

        self.gamma = float(gamma)
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.t_1 = float(t_1)
        self.t_2 = float(t_2)
        self.chi_1 = float(chi_1)
        self.chi_2 = float(chi_2)
        self.T_0 = float(T_0)
        self.T_1 = float(T_1)

        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.t_2 <= self.t_1:
            raise ValueError("t_2 must be greater than t_1")
        if self.chi_1 <= 0.0:
            raise ValueError("chi_1 must be positive")
        if not 0.0 < self.chi_2 < 1.0:
            raise ValueError("chi_2 must lie in (0, 1)")
        if self.T_0 <= 0.0 or self.T_1 <= 0.0:
            raise ValueError("T_0 and T_1 must be positive")

        # Transformed positions p = (rho, vartheta) in R x S^1.
        self.p0 = self.diffeomorphism(self.z0)
        self.p_target = self.diffeomorphism(self.target)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule(seed=theta_seed)
        else:
            self.theta_schedule = theta_schedule

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-9):
        """Integrate flows and apply a mode jump whenever the gap reaches delta."""
        t = self.t_1 if t is None else float(t)
        t_end = self.t_2 if t_end is None else float(t_end)
        if t_end <= t:
            raise ValueError("t_end must be greater than the initial time")

        y = np.r_[self.z0, self.psi0, self.eta0, self.q0]
        solution_segments = []

        while t < t_end - 1e-12:
            y = self._apply_jump(y)

            def q_jump_event(_, event_y):
                return self.delta - self.synergy_gap(
                    self.diffeomorphism(event_y[:2]), self._mode(event_y)
                )

            q_jump_event.terminal = True
            q_jump_event.direction = -1

            segment = solve_ivp(
                fun=self.dynamics,
                t_span=(t, t_end),
                y0=y,
                method="RK45",
                dense_output=True,
                atol=atol,
                rtol=rtol,
                events=q_jump_event,
            )
            solution_segments.append(segment)
            if not segment.success:
                raise RuntimeError(segment.message)
            if segment.status != 1:
                break

            t = float(segment.t[-1])
            y = self.jump_map(segment.y[:, -1])

        return HybridSolution(solution_segments)

    def dynamics(self, t, y):
        """Return the physical dynamics in equation (35)."""
        z = np.asarray(y[:2], dtype=float)
        psi = np.asarray(y[2:4], dtype=float)
        eta = np.asarray(y[4:6], dtype=float)
        q = self._mode(y)

        u_1, u_2 = self.control(z, q, eta)
        z_dot = self.control_gain(t) * u_1 * psi
        psi_dot = u_2 * (self.S @ psi)
        eta_dot = 2.0 * np.pi / (self.T_1 * self.epsilon**2) * (self.S @ eta)
        q_dot = 0.0
        return np.r_[z_dot, psi_dot, eta_dot, q_dot]

    def control(self, z, q, eta):
        """Evaluate the high-frequency controls in equations (38a)--(38b)."""
        p = self.diffeomorphism(z)
        value = self.lyapunov_function(p, q)
        direction = expm(-self.kappa * value * self.S) @ self.e1
        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.T_1 * self.kappa)
        )
        u_1 = gain * float(np.dot(direction, self._unit(eta, "eta")))
        u_2 = 2.0 * np.pi / self.epsilon
        return u_1, u_2

    def diffeomorphism(self, z):
        """Map obstacle-free physical coordinates to R x S^1."""
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = self._vector(z, "z")
            difference = z - self.obstacle_center
            distance = np.linalg.norm(difference)
            if distance <= self.obstacle_radius:
                raise ValueError("z must lie strictly outside the obstacle")
            return np.r_[np.log(distance - self.obstacle_radius), difference / distance]

        if z.ndim != 2 or z.shape[0] != 2:
            raise ValueError("z must have shape (2,) or (2, sample_count)")
        difference = z - self.obstacle_center[:, np.newaxis]
        distance = np.linalg.norm(difference, axis=0)
        if np.any(distance <= self.obstacle_radius):
            raise ValueError("all z samples must lie strictly outside the obstacle")
        return np.vstack(
            (np.log(distance - self.obstacle_radius), difference / distance)
        )

    def diffeomorphism_inverse(self, p):
        """Map R x S^1 coordinates back to physical position."""
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            if p.shape != (3,):
                raise ValueError("p must have shape (3,) or (3, sample_count)")
            unit_vector = self._unit(p[1:], "p[1:]")
            radius = self.obstacle_radius + np.exp(p[0])
            return self.obstacle_center + radius * unit_vector

        if p.ndim != 2 or p.shape[0] != 3:
            raise ValueError("p must have shape (3,) or (3, sample_count)")
        norms = np.linalg.norm(p[1:], axis=0)
        if np.any(norms <= 1e-12):
            raise ValueError("the angular component of p must be nonzero")
        unit_vectors = p[1:] / norms
        radius = self.obstacle_radius + np.exp(p[0])
        return self.obstacle_center[:, np.newaxis] + radius * unit_vectors

    def potential_function(self, unit_vector):
        unit_vector = self._unit(unit_vector, "unit_vector")
        return 1.0 - float(np.dot(self.p_target[1:], unit_vector))

    def sphere_map(self, unit_vector, q):
        unit_vector = self._unit(unit_vector, "unit_vector")
        q = self._validate_mode(q)
        return (
            expm((1.5 - q) * self.potential_function(unit_vector) * self.S)
            @ unit_vector
        )

    def synergistic_potential_function(self, unit_vector, q):
        return self.potential_function(self.sphere_map(unit_vector, q))

    def lyapunov_function(self, p, q):
        p = np.asarray(p, dtype=float)
        if p.shape != (3,):
            raise ValueError("p must have shape (3,)")
        q = self._validate_mode(q)
        rho = p[0]
        rho_target = self.p_target[0]
        radial_difference = np.exp(rho) - np.exp(rho_target)
        return float(
            0.5 * (rho - rho_target) ** 2
            + np.sqrt(radial_difference**2 + 1.0)
            - 1.0
            + self.synergistic_potential_function(p[1:], q)
        )

    def synergy_gap(self, p, q):
        q = self._validate_mode(q)
        values = [self.lyapunov_function(p, candidate) for candidate in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def jump_map(self, y):
        y_plus = np.asarray(y, dtype=float).copy()
        y_plus[-1] = self.argmin_mode(self.diffeomorphism(y_plus[:2]))
        return y_plus

    def _apply_jump(self, y):
        p = self.diffeomorphism(y[:2])
        if self.synergy_gap(p, self._mode(y)) >= self.delta:
            return self.jump_map(y)
        return np.asarray(y, dtype=float).copy()

    def argmin_mode(self, p):
        values = np.array(
            [self.lyapunov_function(p, candidate) for candidate in self.Q]
        )
        return self.Q[int(np.argmin(values))]

    def generate_theta_schedule(
        self, initial_theta=(1.0,), values=(-1.0, 0.0, 1.0), seed=None
    ):
        """Generate a one-channel gain schedule satisfying the dwell constraints."""
        values = tuple(float(value) for value in values)
        theta = tuple(float(value) for value in initial_theta)
        if len(theta) != 1:
            raise ValueError("initial_theta must contain exactly one gain")
        if not values:
            raise ValueError("values must not be empty")

        minimum_dwell = 1.0 / self.chi_1
        schedule = [(self.t_1, theta)]
        time = self.t_1
        monitor = self.T_0
        rng = np.random.default_rng(seed)

        while time < self.t_2 - 1e-12:
            duration = min(minimum_dwell, self.t_2 - time)
            if self._theta_has_zero(theta):
                duration = min(duration, monitor / (1.0 - self.chi_2))
                monitor -= (1.0 - self.chi_2) * duration
            else:
                monitor = min(self.T_0, monitor + self.chi_2 * duration)

            time += duration
            if time >= self.t_2 - 1e-12:
                break

            required_duration = min(minimum_dwell, self.t_2 - time)
            candidates = self._theta_candidates(
                theta, values, required_duration, monitor, self.chi_2
            )
            if not candidates:
                raise ValueError("theta values cannot satisfy the dwell constraints")
            theta = tuple(candidates[rng.integers(len(candidates))])
            schedule.append((time, theta))

        return schedule

    def control_gain(self, t):
        if callable(self.theta_schedule):
            return self._scalar(self.theta_schedule(t), "theta")
        if not self.theta_schedule:
            return 1.0

        value = self.theta_schedule[0][1]
        for switch_time, candidate in self.theta_schedule:
            if t < switch_time:
                break
            value = candidate
        return self._scalar(value, "theta")

    def plot(self, solution):
        """Plot the physical trajectory, obstacle, target, and sampled headings."""
        z = solution.y[:2]
        psi = solution.y[2:4]
        fig, ax = plt.subplots(figsize=(7, 7))
        self._style_axis(fig, ax, z)
        ax.add_patch(self._obstacle_patch())
        ax.plot(z[0], z[1], color=CHARCOAL_THEME["trajectory"], linewidth=2.0)
        ax.scatter(
            *self.target,
            marker="*",
            s=180,
            color=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            zorder=5,
        )
        stride = max(1, z.shape[1] // 18)
        ax.quiver(
            z[0, ::stride],
            z[1, ::stride],
            psi[0, ::stride],
            psi[1, ::stride],
            color=CHARCOAL_THEME["initial"],
            angles="xy",
            scale_units="xy",
            scale=3.0,
            width=0.005,
            zorder=4,
        )
        return fig, ax

    def animate(self, solution, frame_count=360, interval=35, repeat_delay=1200):
        """Animate the nonholonomic trajectory and unknown control direction."""
        frame_count = int(frame_count)
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2")

        times = np.linspace(solution.t[0], solution.t[-1], frame_count)
        states = solution(times)
        z = states[:2]
        psi = states[2:4]
        theta = np.array([[self.control_gain(time)] for time in times])

        fig = plt.figure(figsize=(7, 8), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, height_ratios=(5.0, 1.0))
        ax = fig.add_subplot(grid[0, 0])
        ax_theta = fig.add_subplot(grid[1, 0])
        self._style_axis(fig, ax, z)
        ax.add_patch(self._obstacle_patch())
        ax.plot(
            z[0],
            z[1],
            color=CHARCOAL_THEME["trajectory"],
            linewidth=1.2,
            alpha=0.35,
        )
        ax.scatter(
            *self.target,
            marker="*",
            s=220,
            color=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            zorder=5,
        )

        (trail,) = ax.plot(
            [],
            [],
            color=CHARCOAL_THEME["trajectory"],
            linewidth=2.2,
            zorder=4,
        )
        vehicle = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="#38BDF8",
            edgecolor=CHARCOAL_THEME["edge"],
            linewidth=1.0,
            zorder=6,
        )
        ax.add_patch(vehicle)
        (heading,) = ax.plot(
            [],
            [],
            color=CHARCOAL_THEME["initial"],
            linewidth=2.0,
            zorder=7,
        )
        status = ax.text(
            0.03,
            0.97,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=CHARCOAL_THEME["text"],
            family="monospace",
        )
        control_artists = add_control_state_artists(ax_theta, theta)

        span = max(np.ptp(z[0]), np.ptp(z[1]), 1.0)
        vehicle_scale = 0.035 * span
        body = np.array([[1.4, 0.0], [-0.8, 0.65], [-0.8, -0.65]], dtype=float)

        def update(frame_index):
            center = z[:, frame_index]
            forward = self._unit(psi[:, frame_index], "psi")
            left = np.array([-forward[1], forward[0]])
            rotation = np.column_stack((forward, left))
            vehicle.set_xy(center + vehicle_scale * (body @ rotation.T))
            heading.set_data(
                [center[0], center[0] + 1.8 * vehicle_scale * forward[0]],
                [center[1], center[1] + 1.8 * vehicle_scale * forward[1]],
            )
            trail.set_data(z[0, : frame_index + 1], z[1, : frame_index + 1])
            status.set_text(f"t = {times[frame_index]:.2f}")
            panel = update_control_state_artists(control_artists, theta, frame_index)
            return trail, vehicle, heading, status, *panel

        update(0)
        animation = FuncAnimation(
            fig,
            update,
            frames=np.arange(frame_count),
            interval=interval,
            repeat_delay=repeat_delay,
            blit=True,
        )
        return fig, animation

    def _style_axis(self, fig, ax, z):
        fig.patch.set_facecolor(CHARCOAL_THEME["figure"])
        ax.set_facecolor(CHARCOAL_THEME["axes"])
        for spine in ax.spines.values():
            spine.set_color(CHARCOAL_THEME["grid"])
        ax.tick_params(colors=CHARCOAL_THEME["text"])
        ax.set_title("Nonholonomic Obstacle Avoidance", color=CHARCOAL_THEME["text"])
        ax.set_xlabel(r"$z_1$", color=CHARCOAL_THEME["text"])
        ax.set_ylabel(r"$z_2$", color=CHARCOAL_THEME["text"])
        ax.grid(True, color=CHARCOAL_THEME["grid"], alpha=0.35)
        ax.set_aspect("equal", adjustable="box")

        margin = 0.75
        x_values = np.r_[
            z[0],
            self.target[0],
            self.obstacle_center[0] - self.obstacle_radius,
            self.obstacle_center[0] + self.obstacle_radius,
        ]
        y_values = np.r_[
            z[1],
            self.target[1],
            self.obstacle_center[1] - self.obstacle_radius,
            self.obstacle_center[1] + self.obstacle_radius,
        ]
        ax.set_xlim(x_values.min() - margin, x_values.max() + margin)
        ax.set_ylim(y_values.min() - margin, y_values.max() + margin)

    def _obstacle_patch(self):
        return Circle(
            self.obstacle_center,
            self.obstacle_radius,
            facecolor=CHARCOAL_THEME["grid"],
            edgecolor=CHARCOAL_THEME["text"],
            linewidth=1.2,
            zorder=3,
        )

    @classmethod
    def _theta_candidates(cls, theta, values, required_duration, monitor, chi_2):
        candidates = [
            candidate
            for candidate in cls._theta_space(values, len(theta))
            if candidate != theta
        ]
        if monitor < (1.0 - chi_2) * required_duration - 1e-12:
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
        tails = Nonholonomic._theta_space(values, dimension - 1)
        return [(value, *tail) for value in values for tail in tails]

    @staticmethod
    def _theta_has_zero(theta):
        return any(np.isclose(value, 0.0) for value in theta)

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        return vector.copy()

    def _outside_obstacle(self, value, name):
        vector = self._vector(value, name)
        if np.linalg.norm(vector - self.obstacle_center) <= self.obstacle_radius:
            raise ValueError(f"{name} must lie strictly outside the obstacle")
        return vector

    @classmethod
    def _unit(cls, value, name):
        vector = cls._vector(value, name)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError(f"{name} must be nonzero")
        return vector / norm

    @staticmethod
    def _scalar(value, name):
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size != 1 or not np.isfinite(array[0]):
            raise ValueError(f"{name} must be a finite scalar")
        return float(array[0])

    def _validate_mode(self, q):
        q_value = self._scalar(q, "q")
        mode = int(round(q_value))
        if not np.isclose(q_value, mode) or mode not in self.Q:
            raise ValueError(f"q must be one of {self.Q}")
        return mode

    def _mode(self, y):
        return self._validate_mode(y[-1])


# Preserve the public name used by earlier versions of this repository.
Nonholomonic = Nonholonomic
