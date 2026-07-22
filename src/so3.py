import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from charcoal_animation import (
    CHARCOAL_THEME,
    add_control_state_artists,
    create_drone_artists_3d,
    create_sphere_animation_figure,
    position_drone_artists_3d,
    update_control_state_artists,
)
from hybrid_solution import HybridSolution


class SO3:
    """Simulation of the SO(3) example using column-wise vectorization."""

    Q = (1, 2)
    S = np.array([[0.0, 1.0], [-1.0, 0.0]])
    e1 = np.array([1.0, 0.0])

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
        angular_velocity,
        T_0=1.0,
        theta_schedule=None,
        theta_seed=None,
    ):
        self.R0 = self._rotation_matrix(p0, "p0")
        self.R_target = self._rotation_matrix(target, "target")
        self.p0 = self.diffeomorphism(self.R0)
        self.target = self.diffeomorphism(self.R_target)
        self.q0 = self._mode_value(q0)

        eta0 = np.asarray(eta0, dtype=float)
        if eta0.size != 6:
            raise ValueError("eta0 must contain three two-dimensional oscillators")
        self.eta0 = self._unit_rows(eta0.reshape(3, 2)).reshape(-1)

        self.gamma = float(gamma)
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.t_1 = float(t_1)
        self.t_2 = float(t_2)
        self.chi_1 = float(chi_1)
        self.chi_2 = float(chi_2)
        self.T_0 = float(T_0)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.chi_1 <= 0.0:
            raise ValueError("chi_1 must be positive")
        if not 0.0 < self.chi_2 < 1.0:
            raise ValueError("chi_2 must lie strictly between zero and one")

        self.control_gain_constants = np.asarray(
            control_gain_constants, dtype=float
        ).reshape(-1)
        if self.control_gain_constants.shape != (3,):
            raise ValueError("control_gain_constants must contain three periods")
        if np.any(self.control_gain_constants <= 0.0):
            raise ValueError("oscillator periods must be positive")

        self.angular_velocity = np.asarray(angular_velocity, dtype=float).reshape(-1)
        if self.angular_velocity.shape != (3,):
            raise ValueError("angular_velocity must have length three")
        if np.any(self.angular_velocity <= 0.0):
            raise ValueError("angular_velocity entries must be positive")

        weight_sum = float(np.sum(self.angular_velocity))
        self.A = 3.0 * np.diag(self.angular_velocity) / weight_sum
        self.omega = self._unit(self.angular_velocity)
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

            def jump_event(_, event_y):
                p = event_y[:9]
                q = self._mode(event_y)
                return self.delta - self.synergy_gap(p, q)

            jump_event.terminal = True
            jump_event.direction = -1

            sol = solve_ivp(
                fun=self.dynamics,
                t_span=(t, t_end),
                y0=y,
                method="RK45",
                dense_output=True,
                atol=atol,
                rtol=rtol,
                events=jump_event,
            )
            solution_segments.append(sol)

            if sol.status != 1:
                break

            t = float(sol.t[-1])
            y = self.jump_map(sol.y[:, -1])

        return HybridSolution(solution_segments)

    def dynamics(self, t, y):
        R = self._matrix(y[:9])
        eta = y[9:-1].reshape(3, 2)

        theta = self._get_control_gain(t)
        u = self.control(y)
        R_dot = R @ self._skew_symmetric(theta * u)

        frequencies = 2.0 * np.pi * self.control_gain_constants**-1 * self.epsilon**-2
        eta_dot = frequencies[:, np.newaxis] * (self.S @ eta.T).T
        q_dot = 0.0

        return np.r_[R_dot.reshape(-1, order="F"), eta_dot.reshape(-1), q_dot]

    @staticmethod
    def diffeomorphism(R):
        array = np.asarray(R, dtype=float)
        if array.shape == (3, 3):
            matrix = array
        elif array.size == 9:
            matrix = array.reshape(3, 3, order="F")
        else:
            raise ValueError("R must be a 3-by-3 matrix or length-9 vector")
        return matrix.reshape(-1, order="F")

    def lyapunov_function_so3(self, R, q):
        warped_R = self.synergistic_potential_function(R, q)
        return self.potential_function(warped_R)

    def lyapunov_function(self, p, q):
        return self.lyapunov_function_so3(self._matrix(p), self._mode_value(q))

    def control(self, y):
        p = y[:9]
        eta = self._unit_rows(y[9:-1].reshape(3, 2))
        q = self._mode(y)
        V = self.lyapunov_function(p, q)
        direction = expm(-self.kappa * V * self.S) @ self.e1
        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.control_gain_constants * self.kappa)
        )
        return gain * (eta @ direction)

    def potential_function(self, R):
        attitude_error = self.R_target.T @ self._matrix(R)
        return float(np.trace(self.A @ (np.eye(3) - attitude_error)))

    def synergistic_potential_function(self, R, q):
        R = self._matrix(R)
        attitude_error = self.R_target.T @ R
        base_potential = float(np.trace(self.A @ (np.eye(3) - attitude_error)))
        warped_error = (
            expm(
                ((3.0 - 2.0 * q) / 4.0)
                * base_potential
                * self._skew_symmetric(self.omega)
            )
            @ attitude_error
        )
        return self.R_target @ warped_error

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float, copy=True)
        R = self._project_so3(self._matrix(y_plus[:9]))
        eta = y_plus[9:-1].reshape(3, 2)
        y_plus[:9] = self.diffeomorphism(R)
        y_plus[9:-1] = self._unit_rows(eta).reshape(-1)
        y_plus[-1] = self.argmin_mode(y_plus[:9])
        return y_plus

    def control_vector_fields(self, p, idx):
        e_i_skew = self._skew_symmetric(np.eye(3)[idx])
        return -np.kron(e_i_skew, np.eye(3)) @ np.asarray(p, dtype=float)

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values) + 1)

    def generate_theta_schedule(self, initial_theta=(1.0, 1.0, 1.0), seed=None):
        values = self.values
        theta = tuple(float(value) for value in initial_theta)
        min_dwell = 1.0 / self.chi_1
        schedule = [(self.t_1, theta)]
        t = self.t_1
        monitor = self.T_0
        rng = np.random.default_rng(seed)

        while t < self.t_2 - 1e-12:
            remaining = self.t_2 - t
            duration = min(min_dwell, remaining)

            if self._theta_has_zero(theta):
                duration = min(duration, monitor / (1.0 - self.chi_2))
                monitor -= (1.0 - self.chi_2) * duration
            else:
                monitor = min(self.T_0, monitor + self.chi_2 * duration)

            t += duration
            if t >= self.t_2 - 1e-12:
                break

            required_duration = min(min_dwell, self.t_2 - t)
            candidates = self._theta_candidates(
                theta, values, required_duration, monitor, self.chi_2
            )
            theta = tuple(candidates[rng.integers(len(candidates))])
            schedule.append((t, theta))

        return schedule

    def _get_control_gain(self, t):
        if callable(self.theta_schedule):
            return self._control_gain_vector(self.theta_schedule(t))

        theta = self.theta_schedule[0][1]
        for switch_time, candidate in self.theta_schedule:
            if t < switch_time:
                break
            theta = candidate
        return self._control_gain_vector(theta)

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
        attitudes = [
            self._project_so3(self._matrix(states[:9, index]))
            for index in range(frame_count)
        ]
        gains = np.vstack([self._get_control_gain(t) for t in times])

        fig, ax, ax_control = create_sphere_animation_figure(r"$SO(3)$  Stabilization")
        target_direction = self.R_target[:, 2]
        ax.scatter(
            *target_direction,
            color=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            marker="*",
            s=150,
            label="Target",
            zorder=6,
        )
        target_drone = create_drone_artists_3d(
            ax,
            body_color=CHARCOAL_THEME["target"],
            rotor_color=CHARCOAL_THEME["target"],
            nose_color=CHARCOAL_THEME["target"],
            alpha=0.3,
        )
        position_drone_artists_3d(target_drone, 1.08 * target_direction, self.R_target)

        initial_attitude = attitudes[0]
        initial_direction = initial_attitude[:, 2]
        ax.scatter(
            *initial_direction,
            color=CHARCOAL_THEME["initial"],
            edgecolor=CHARCOAL_THEME["edge"],
            s=42,
            label="Start",
            zorder=5,
        )
        drone = create_drone_artists_3d(ax)
        status = ax.text2D(
            0.03,
            0.96,
            "",
            transform=ax.transAxes,
            color=CHARCOAL_THEME["text"],
            va="top",
        )
        legend = ax.legend(loc="upper right", frameon=False)
        for text in legend.get_texts():
            text.set_color(CHARCOAL_THEME["text"])

        control_artists = add_control_state_artists(ax_control, gains)

        def update(frame_index):
            attitude = attitudes[frame_index]
            pointing = attitude[:, 2]
            drone_artists = position_drone_artists_3d(drone, 1.08 * pointing, attitude)

            mode = self._mode(states[:, frame_index])
            status.set_text(f"t = {times[frame_index]:.2f}")
            panel_artists = update_control_state_artists(
                control_artists, gains, frame_index
            )
            return (
                *drone_artists,
                status,
                *panel_artists,
            )

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

    def render_mujoco(self, solution, output_path="Animations/so3_mujoco.mp4", **kwargs):
        """Render a solved trajectory with the MuJoCo quadrotor scene."""
        from so3_mujoco import render_so3_animation

        return render_so3_animation(self, solution, output_path, **kwargs)

    def _apply_jump(self, y):
        p = y[:9]
        q = self._mode(y)
        if self.synergy_gap(p, q) >= self.delta:
            return self.jump_map(y)
        return np.array(y, dtype=float, copy=True)

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
        tails = SO3._theta_space(values, dimension - 1)
        return [(value, *tail) for value in values for tail in tails]

    @staticmethod
    def _theta_has_zero(theta):
        return any(np.isclose(value, 0.0) for value in theta)

    @staticmethod
    def _matrix(value):
        array = np.asarray(value, dtype=float)
        if array.shape == (3, 3):
            return array
        if array.size != 9:
            raise ValueError("an attitude must be a 3-by-3 matrix or length-9 vector")
        return array.reshape(3, 3, order="F")

    @classmethod
    def _rotation_matrix(cls, value, name):
        R = cls._matrix(value)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
            raise ValueError(f"{name} must be orthogonal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
            raise ValueError(f"{name} must have determinant +1")
        return np.array(R, dtype=float, copy=True)

    @staticmethod
    def _project_so3(R):
        U, _, Vt = np.linalg.svd(R)
        correction = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
        return U @ correction @ Vt

    @staticmethod
    def _skew_symmetric(arr):
        a, b, c = np.asarray(arr, dtype=float)
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
