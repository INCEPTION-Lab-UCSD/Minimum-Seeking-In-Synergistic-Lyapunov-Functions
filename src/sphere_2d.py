from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from charcoal_animation import CHARCOAL_THEME
from hybrid_solution import HybridSolution


class Sphere_2D:
    """
    Simulation of the S1 example from Section 4.1.

    State ordering is [p1, p2, q, eta1, eta2], where p and eta lie on S1
    and q selects one of the two synergistic potentials.
    """

    Q = (1, 2)
    S = np.array([[0.0, 1.0], [-1.0, 0.0]])
    e1 = np.array([1.0, 0.0])

    def __init__(
        self,
        p0,
        q0,
        p_target,
        gamma,
        kappa,
        epsilon,
        t_1,
        t_2,
        eta0,
        theta_schedule=None,
        T=1.0,
        delta=0.25,
        max_step=0.01,
        chi_1=0.5,
        chi_2=0.5,
        T_0=1.0,
        theta_seed=None,
    ):
        self.p0 = self._unit(p0)
        self.q0 = float(np.asarray(q0).reshape(-1)[0])
        self.p_target = self._unit(p_target)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.t_1 = float(t_1)
        self.t_2 = float(t_2)
        self.eta0 = self._unit(eta0)
        self.T = float(T)
        self.delta = float(delta)
        self.max_step = float(max_step)
        self.chi_1 = float(chi_1)
        self.chi_2 = float(chi_2)
        self.T_0 = float(T_0)
        self.theta_schedule = theta_schedule
        self.r = int(len(p0) / 2)
        self.Epsilon = list(product([1, 0, -1], repeat=self.r))
        if self.theta_schedule is None:
            self.theta_schedule = self.generate_theta_schedule(
                self.t_1,
                self.t_2,
                self.chi_1,
                self.chi_2,
                self.T_0,
                seed=theta_seed,
            )

    def potential_function(self, p):
        return 1.0 - float(np.dot(self.p_target, p))

    def sphere_map(self, p, q):
        return expm((1.5 - q) * self.potential_function(p) * self.S) @ p

    def synergistic_potential_function(self, p, q):
        return self.potential_function(self.sphere_map(p, q))

    def solve(self, t=None, t_end=None, rtol=1e-6, atol=1e-8):
        if t is None:
            t = self.t_1
        if t_end is None:
            t_end = self.t_2

        y = np.r_[self.p0, self.q0, self.eta0]
        solution_segments = []

        while t < t_end - 1e-12:
            y = self._apply_jump_if_needed(y)

            def q_jump_event(_, event_y):
                p = self._unit(event_y[:2])
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
                max_step=self.max_step,
            )
            solution_segments.append(sol)

            if sol.status != 1:
                break

            t = float(sol.t[-1])
            y = self.jump_map(sol.y[:, -1])

        return HybridSolution(solution_segments)

    def dynamics(self, t, y):
        p = self._unit(y[:2])
        q = self._mode(y)
        eta = self._unit(y[3:5])

        u = self.control(p, q, eta)
        b = self.control_vector_fields(p)
        theta = self.control_gain(t)

        p_dot = b * theta * u
        q_dot = 0.0
        eta_dot = 2.0 * np.pi * self.T**-1 * self.epsilon**-2 * (self.S @ eta)

        return np.r_[p_dot, q_dot, eta_dot]

    def lyapunov_function(self, p, q):
        return self.synergistic_potential_function(p, q)

    def control_vector_fields(self, p):
        return self.S @ p

    def jump_map(self, y):
        y_plus = np.array(y, dtype=float, copy=True)
        p = y_plus[:2]
        q = y_plus[2]
        eta = y_plus[3:]
        y_plus[:2] = self._unit(p)
        y_plus[2] = self.argmin_mode(p)
        y_plus[3:] = self._unit(eta)
        return y_plus

    def control_gain(self, t):
        if callable(self.theta_schedule):
            return float(self.theta_schedule(t))

        if self.theta_schedule is not None:
            value = self.theta_schedule[0][1]
            for switch_time, candidate in self.theta_schedule:
                if t < switch_time:
                    break
                value = candidate
            return float(value)

        return self._default_control_gain(t)

    def control(self, p, q, eta):
        v = self.lyapunov_function(p, q)
        direction = expm(-self.kappa * v * self.S) @ self.e1
        gain = self.epsilon**-1 * np.sqrt(
            (4.0 * np.pi * self.gamma) / (self.T * self.kappa)
        )
        return gain * float(np.dot(direction, eta))

    def synergy_gap(self, p, q):
        values = [self.lyapunov_function(p, q_i) for q_i in self.Q]
        return self.lyapunov_function(p, q) - min(values)

    def argmin_mode(self, p):
        values = np.array([self.lyapunov_function(p, q) for q in self.Q])
        return int(np.argmin(values) + 1)

    @staticmethod
    def generate_theta_schedule(
        t_start,
        t_end,
        chi_1,
        chi_2,
        T_0,
        initial_theta=1.0,
        values=(-1.0, 0.0, 1.0),
        seed=None,
    ):
        """
        Generate a piecewise-constant theta schedule satisfying the bounds

            N#(t1, t2) <= chi_1 (t2 - t1) + 1,
            T#(t1, t2) <= chi_2 (t2 - t1) + T_0.

        The first bound is enforced by spacing switches at least 1 / chi_1
        apart. The second bound is enforced with the monitor from (14c): zero
        control consumes budget at rate 1 - chi_2, while nonzero control
        replenishes budget at rate chi_2 up to T_0.
        """
        t_start = float(t_start)
        t_end = float(t_end)
        chi_1 = float(chi_1)
        chi_2 = float(chi_2)
        T_0 = float(T_0)
        values = tuple(float(value) for value in values)
        initial_theta = float(initial_theta)

        min_dwell = 1.0 / chi_1
        schedule = [(t_start, initial_theta)]
        t = t_start
        theta = initial_theta
        last_nonzero = initial_theta if initial_theta != 0.0 else 1.0
        monitor = T_0
        rng = np.random.default_rng(seed) if seed is not None else None

        while t < t_end - 1e-12:
            remaining = t_end - t
            segment_floor = min(min_dwell, remaining)
            segment_ceiling = remaining

            if theta == 0.0 and chi_2 < 1.0:
                segment_ceiling = min(segment_ceiling, monitor / (1.0 - chi_2))

            if segment_ceiling < segment_floor - 1e-12:
                theta = -last_nonzero
                schedule.append((t, theta))
                continue

            if remaining <= min_dwell + 1e-12:
                duration = remaining
            elif rng is None:
                duration = segment_floor
            else:
                duration = rng.uniform(segment_floor, segment_ceiling)

            if theta == 0.0:
                monitor -= (1.0 - chi_2) * duration
            else:
                monitor = min(T_0, monitor + chi_2 * duration)
                last_nonzero = theta

            t += duration
            if t >= t_end - 1e-12:
                break

            candidates = Sphere_2D._theta_candidates(
                theta, last_nonzero, values, min(min_dwell, t_end - t), monitor, chi_2
            )
            theta = float(rng.choice(candidates)) if rng is not None else candidates[0]
            schedule.append((t, theta))

        return schedule

    @staticmethod
    def _theta_candidates(
        theta, last_nonzero, values, required_duration, monitor, chi_2
    ):
        if theta == 0.0:
            preferred = (-last_nonzero, last_nonzero)
        else:
            preferred = (0.0, -theta)

        candidates = [
            value for value in preferred if value in values and value != theta
        ]
        candidates.extend(
            value for value in values if value != theta and value not in candidates
        )

        if chi_2 < 1.0:
            zero_budget_required = (1.0 - chi_2) * required_duration
            if monitor < zero_budget_required - 1e-12:
                candidates = [value for value in candidates if value != 0.0]

        return candidates

    def plot(self, solution):
        t = solution.t
        p_1 = solution.y[0]
        p_2 = solution.y[1]
        theta = np.array([self.control_gain(t_i) for t_i in t])

        fig = plt.figure(figsize=(10, 4))
        ax_unit_circle = fig.add_subplot()

        theta = np.linspace(0.0, 2 * np.pi, 200)

        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax_unit_circle.plot(circle_x, circle_y, color="black")
        ax_unit_circle.scatter(p_1[-1], p_2[-1], color="red")

        return fig, ax_unit_circle

    def animate(self, solution, frame_step=25, interval=40, repeat_delay=1200):
        frame_step = int(frame_step)
        if frame_step < 1:
            raise ValueError("frame_step must be a positive integer")

        frames = np.arange(0, len(solution.t), frame_step)
        if frames[-1] != len(solution.t) - 1:
            frames = np.r_[frames, len(solution.t) - 1]

        fig, ax_unit_circle = plt.subplots(
            figsize=(7, 7), facecolor=CHARCOAL_THEME["figure"]
        )
        artists = self._setup_unit_circle_ax(ax_unit_circle)

        def update(frame_idx):
            t = solution.t[frame_idx]
            y = solution.y[:, frame_idx]
            p = self._unit(y[:2])
            q = self._mode(y)

            artists["point"].set_data([p[0]], [p[1]])
            gain = self.control_gain(t)
            gain_name, gain_color = self._control_gain_visual(gain)
            artists["status"].set_text(f"t = {t:.2f}\nq = {q}")
            artists["control_gain"].set_text(f"CONTROL GAIN\n{gain_name}  {gain:+.0f}")
            artists["control_gain"].set_color(gain_color)
            return tuple(artists.values())

        update(frames[0])
        animation = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            repeat_delay=repeat_delay,
        )
        fig.tight_layout()
        return fig, animation

    def _setup_unit_circle_ax(self, ax, point_color=None):
        point_color = point_color or CHARCOAL_THEME["trajectory"]
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.set_facecolor(CHARCOAL_THEME["axes"])
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            color=CHARCOAL_THEME["text"],
            linewidth=1.5,
        )
        ax.scatter(
            self.p_target[0],
            self.p_target[1],
            color=CHARCOAL_THEME["target"],
            edgecolor=CHARCOAL_THEME["edge"],
            marker="*",
            s=200,
            label="target",
            zorder=4,
        )

        ax.scatter(
            self.p0[0],
            self.p0[1],
            color=CHARCOAL_THEME["initial"],
            marker="o",
            s=200,
            label="start",
            zorder=4,
        )

        (point,) = ax.plot(
            [],
            [],
            marker="o",
            color=point_color,
            markersize=9,
            linestyle="none",
            label="point",
            zorder=5,
        )
        status = ax.text(
            0.03,
            0.97,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            color=CHARCOAL_THEME["text"],
        )
        control_gain = ax.text(
            0.5,
            0.5,
            "",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            linespacing=1.35,
            bbox={
                "facecolor": CHARCOAL_THEME["axes"],
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 6,
            },
            zorder=6,
        )

        ax.set_title("Circle Stabilization", color=CHARCOAL_THEME["text"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(False)
        legend = ax.legend(loc="lower right", frameon=False)
        for text in legend.get_texts():
            text.set_color(CHARCOAL_THEME["text"])

        return {
            "point": point,
            "status": status,
            "control_gain": control_gain,
        }

    @staticmethod
    def _control_gain_visual(gain):
        if np.isclose(gain, 0.0):
            return "BLIND", CHARCOAL_THEME["initial"]
        if gain > 0.0:
            return "NORMAL", CHARCOAL_THEME["target"]
        return "REVERSED", CHARCOAL_THEME["trajectory"]

    def _apply_jump_if_needed(self, y):
        if self.synergy_gap(y[:2], self._mode(y)) >= self.delta:
            return self.jump_map(y)
        return y

    def _default_control_gain(self, t):
        return 1.0

    def _mode(self, y):
        return int(round(float(y[2])))

    def _unit(self, vector):
        vector = np.asarray(vector, dtype=float)
        return vector / np.linalg.norm(vector)
