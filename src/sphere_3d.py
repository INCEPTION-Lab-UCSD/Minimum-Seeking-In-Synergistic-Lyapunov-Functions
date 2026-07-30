import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from charcoal_animation import (
    CHARCOAL_THEME,
    add_control_state_artists,
    add_trajectory_artists,
    create_drone_artists_3d,
    create_sphere_animation_figure,
    position_drone_artists_3d,
    update_control_state_artists,
    update_trajectory_artists,
)
from hybrid_solution import HybridSolution


class Sphere_3D:
    """Simulation of the S2 example with three two-state oscillators."""

    Q = (1, 2)
    e1 = np.array([1, 0, 0], dtype=float)
    e2 = np.array([0, 1, 0], dtype=float)
    e3 = np.array([0, 0, 1], dtype=float)
    S = np.array([[0, 1], [-1, 0]], dtype=float)

    def __init__(
        self,
        p0,
        eta0,
        q0,
        target,
        gamma,
        delta,
        kappa,
        epsilon,
        t_1,
        t_2,
        chi_1,
        chi_2,
        control_gain_constants,
        T_0=1.0,
        theta_schedule=None,
        theta_seed=None,
    ):

        self.p0 = self._unit(p0)
        self.target = self._unit(target)
        self.q0 = self._mode_value(q0)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.t_1 = float(t_1)
        self.t_2 = float(t_2)
        self.chi_1 = float(chi_1)
        self.chi_2 = float(chi_2)
        self.T_0 = float(T_0)
        self.control_gain_constants = np.asarray(
            control_gain_constants, dtype=float
        ).reshape(-1)
        if self.control_gain_constants.shape != (3,):
            raise ValueError("control_gain_constants must contain three periods")
        if np.any(self.control_gain_constants <= 0.0):
            raise ValueError("oscillator periods must be positive")

        eta0 = np.asarray(eta0, dtype=float)
        if eta0.size != 6:
            raise ValueError("eta0 must contain three two-dimensional oscillators")
        self.eta0 = self._unit_rows(eta0.reshape(3, 2)).reshape(-1)

        axis = np.eye(3)[np.argmin(np.abs(self.target))]
        self.target_orthogonal = self._unit(np.cross(self.target, axis))
        self.values = (-1.0, 0.0, 1.0)

        if theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule(seed=theta_seed)
        else:
            self.theta_schedule = theta_schedule

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-9):
        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2

        y = np.r_[self.p0, self.eta0, self.q0]
        solution_segments = []

        while t < t_end - 1e-12:
            y = self._apply_jump(y)

            def q_jump_event(_, event_y):
                p = self._unit(event_y[:3])
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

    def dynamics(self, t, y):
        p = self._unit(y[:3])
        eta = y[3:-1].reshape(3, 2)
        q = self._mode(y)

        theta = self.get_control_gain(t)
        u = self.control(p, q, self._unit_rows(eta))
        vector_fields = np.column_stack(
            [self.control_vector_fields(p, idx) for idx in range(3)]
        )
        p_dot = vector_fields @ (theta * u)

        frequencies = 2.0 * np.pi * self.control_gain_constants**-1 * self.epsilon**-2
        eta_dot = frequencies[:, np.newaxis] * (self.S @ eta.T).T
        q_dot = 0.0

        return np.r_[p_dot, eta_dot.reshape(-1), q_dot]

    def get_control_gain(self, t):
        if callable(self.theta_schedule):
            return self._control_gain_vector(self.theta_schedule(t))

        theta = self.theta_schedule[0][1]
        for switch_time, candidate in self.theta_schedule:
            if t < switch_time:
                break
            theta = candidate
        return self._control_gain_vector(theta)

    def lyapunov_function(self, p, q):
        return self.potential_function(self.synergistic_potential_function(p, q))

    def potential_function(self, p):
        return 1.0 - float(np.dot(p, self.target))

    def synergistic_potential_function(self, p, q):
        rotation = expm(
            (1.5 - q)
            * self.potential_function(p)
            * self._skew_symmetric(self.target_orthogonal)
        )
        return rotation @ p

    def control_vector_fields(self, p, idx):
        e_i = np.eye(3)[idx]
        return e_i - np.dot(p, e_i) * p

    def control(self, p, q, eta):
        V = self.lyapunov_function(p, q)
        e1 = np.array([1, 0], dtype=float)

        direction = expm(-self.kappa * V * self.S) @ e1

        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.control_gain_constants * self.kappa)
        )

        return gain * (eta @ direction)

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float).copy()
        p = y_plus[:3]
        eta = y_plus[3:-1].reshape(3, 2)
        y_plus[:3] = self._unit(p)
        y_plus[3:-1] = self._unit_rows(eta).reshape(-1)
        y_plus[-1] = self.argmin_mode(p)

        return y_plus

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values)) + 1

    def generate_theta_schedule(self, initial_theta=(1.0, 1.0, 1.0), seed=None):
        values = self.values
        t_start = self.t_1
        t_end = self.t_2
        chi_1 = self.chi_1
        chi_2 = self.chi_2
        T_0 = self.T_0

        theta = initial_theta

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

    def animate(
        self,
        solution,
        frame_count=240,
        interval=40,
        repeat_delay=1200,
    ):
        t_start = max(self.t_1, float(solution.t[0]))
        t_end = min(self.t_2, float(solution.t[-1]))
        times = np.linspace(t_start, t_end, frame_count)
        states = solution(times)
        directions = np.column_stack(
            [self._unit(states[:3, index]) for index in range(frame_count)]
        ).T
        attitudes = self._vehicle_attitudes(directions)
        gains = np.vstack([self.get_control_gain(t) for t in times])

        fig, ax, ax_trajectory, ax_control = create_sphere_animation_figure(
            r"$S^2$ Stabilization"
        )
        start = directions[0]
        ax.scatter(
            *start,
            color=CHARCOAL_THEME["initial"],
            edgecolor=CHARCOAL_THEME["edge"],
            s=45,
            zorder=5,
        )
        ax.scatter(
            *self.target,
            color=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            marker="*",
            s=150,
            zorder=6,
        )
        drone = create_drone_artists_3d(ax, scale=0.15)
        status = ax.text2D(
            0.03,
            0.96,
            "",
            transform=ax.transAxes,
            color=CHARCOAL_THEME["text"],
            va="top",
            fontsize=12,
        )
        legend = ax.legend(loc="upper right", frameon=False)
        for text in legend.get_texts():
            text.set_color(CHARCOAL_THEME["text"])

        control_artists = add_control_state_artists(ax_control, gains)
        trajectory_artists = add_trajectory_artists(
            ax_trajectory,
            times,
            directions,
            (r"$p_1$", r"$p_2$", r"$p_3$"),
        )

        def update(frame_index):
            direction = directions[frame_index]
            drone_artists = position_drone_artists_3d(
                drone,
                1.04 * direction,
                attitudes[frame_index],
            )

            status.set_text(f"t = {times[frame_index]:.2f}")
            panel_artists = update_control_state_artists(
                control_artists, gains, frame_index
            )
            path_artists = update_trajectory_artists(
                trajectory_artists, times, directions, frame_index
            )
            return *drone_artists, status, *path_artists, *panel_artists

        animation = FuncAnimation(
            fig,
            update,
            frames=frame_count,
            interval=interval,
            repeat_delay=repeat_delay,
            blit=False,
        )
        update(0)
        return fig, animation

    @classmethod
    def _vehicle_attitudes(cls, directions):
        attitudes = []
        previous_forward = None

        for index, radial in enumerate(directions):
            previous_index = max(0, index - 1)
            next_index = min(len(directions) - 1, index + 1)
            forward = directions[next_index] - directions[previous_index]
            forward -= np.dot(forward, radial) * radial

            if np.linalg.norm(forward) < 1e-7 and previous_forward is not None:
                forward = previous_forward - np.dot(previous_forward, radial) * radial
            if np.linalg.norm(forward) < 1e-7:
                reference = np.eye(3)[np.argmin(np.abs(radial))]
                forward = reference - np.dot(reference, radial) * radial

            forward = cls._unit(forward)
            lateral = cls._unit(np.cross(radial, forward))
            forward = cls._unit(np.cross(lateral, radial))
            attitudes.append(np.column_stack((forward, lateral, radial)))
            previous_forward = forward

        return attitudes

    def _apply_jump(self, y):
        p = self._unit(y[:3])
        q = self._mode(y)
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return y

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

        tails = Sphere_3D._theta_space(values, dimension - 1)
        return [(value, *tail) for value in values for tail in tails]

    @staticmethod
    def _theta_has_zero(theta):
        return np.any([np.isclose(value, 0.0) for value in theta])

    @staticmethod
    def _skew_symmetric(arr):
        a, b, c = arr
        return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])

    @staticmethod
    def _unit(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            raise ValueError("cannot normalize a zero vector")
        return vector / norm

    @classmethod
    def _unit_rows(cls, vectors):
        return np.vstack([cls._unit(vector) for vector in vectors])

    @classmethod
    def _mode_value(cls, q):
        mode = int(round(float(np.asarray(q).reshape(-1)[0])))
        if mode not in cls.Q:
            raise ValueError(f"q must be one of {cls.Q}")
        return mode

    @classmethod
    def _mode(cls, y):
        return cls._mode_value(y[-1])

    @staticmethod
    def _control_gain_vector(theta):
        vector = np.asarray(theta, dtype=float).reshape(-1)
        if vector.size == 1:
            return np.full(3, vector[0], dtype=float)
        if vector.size != 3:
            raise ValueError("theta must be scalar or length 3")
        return vector
