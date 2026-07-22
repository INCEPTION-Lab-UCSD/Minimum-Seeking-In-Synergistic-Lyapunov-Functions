import matplotlib.pyplot as plt
import numpy as np
import scipy

import hybrid_solution


class Nonholomonic:
    Q = (1, 2)
    S = np.array([[0, 1], [-1, 0]], dtype=float)
    e1 = np.array([1, 0], dtype=float)

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

        self.p0 = p0
        self.eta0 = eta0
        self.q0 = q0
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.t_1 = float(t_1)
        self.t_2 = float(t_2)
        self.chi_1 = float(chi_1)
        self.chi_2 = float(chi_2)
        self.T_0 = float(T_0)

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

            sol = scipy.integrate.solve_ivp(
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

    def control(self, x, eta):
        lyapunov = self.lyapunov_function(x)
        u_1 = (
            1
            / self.epsilon
            * np.sqrt((4 * np.pi * self.gamma) / self.kappa)
            * np.dot(
                scipy.linalg.expm(-self.kappa * lyapunov @ self.S) @ self.e1, eta[0]
            )
        )

        u_2 = 2 * np.pi * 1 / self.epsilon

        return u_1, u_2

    def lyapunov_function(self, x):
        rho = x[0]
        nu = x[1:3]
        q = x[-1]
        return 0

    def _get_control_gain(self):
        pass

    def _mode(self):
        pass

    def _skew_symmetric(self):
        pass
