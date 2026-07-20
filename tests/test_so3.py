import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from so3 import SO3


class SO3Test(unittest.TestCase):
    def make_simulation(self, p0=None):
        if p0 is None:
            p0 = np.diag([-1.0, 1.0, -1.0])
        return SO3(
            p0=p0,
            eta0=np.tile([1.0, 0.0], (3, 1)),
            q0=1,
            target=np.eye(3),
            gamma=1.0,
            delta=0.2,
            kappa=4.0,
            epsilon=1.0 / np.sqrt(12.0 * np.pi),
            t_1=0.0,
            t_2=0.1,
            chi_1=1.0,
            chi_2=0.5,
            control_gain_constants=np.array([1.0, 2.0, 3.0]),
            angular_velocity=np.array([11.0, 12.0, 13.0]),
            theta_schedule=[(0.0, (1.0, 1.0, 1.0))],
        )

    def test_column_vectorization_and_control_vector_fields(self):
        simulation = self.make_simulation()
        self.assertTrue(
            np.allclose(
                simulation.diffeomorphism(simulation.p0), simulation.p0
            )
        )

        for idx in range(3):
            field = simulation.control_vector_fields(simulation.p0, idx)
            field_matrix = field.reshape(3, 3, order="F")
            expected = simulation.R0 @ simulation._skew_symmetric(np.eye(3)[idx])
            self.assertTrue(np.allclose(field_matrix, expected))

    def test_target_potential_and_control_dimensions(self):
        simulation = self.make_simulation()
        target_p = simulation.diffeomorphism(simulation.R_target)
        self.assertAlmostEqual(simulation.lyapunov_function(target_p, 1), 0.0)
        self.assertAlmostEqual(simulation.lyapunov_function(target_p, 2), 0.0)

        state = np.r_[simulation.p0, simulation.eta0, simulation.q0]
        self.assertEqual(simulation.control(state).shape, (3,))
        self.assertEqual(simulation.dynamics(0.0, state).shape, (16,))

    def test_reflection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "determinant"):
            self.make_simulation(-np.eye(3))

    def test_jump_selects_lower_potential_mode(self):
        simulation = self.make_simulation()
        R = np.array(
            [
                [0.07984482, -0.17415087, 0.98147658],
                [0.98654660, -0.12710547, -0.10281057],
                [0.14265559, 0.97648127, 0.16165923],
            ]
        )
        R = simulation._project_so3(R)
        state = np.r_[simulation.diffeomorphism(R), simulation.eta0, 1]
        jumped = simulation._apply_jump(state)

        self.assertGreater(simulation.synergy_gap(state[:9], 1), simulation.delta)
        self.assertEqual(int(jumped[-1]), 2)
        self.assertEqual(int(state[-1]), 1)

    def test_short_solution_remains_on_so3(self):
        simulation = self.make_simulation()
        solution = simulation.solve()

        self.assertAlmostEqual(solution.t[-1], simulation.t_2)
        self.assertTrue(np.all(np.isfinite(solution.y)))
        for p in solution.y[:9].T:
            R = p.reshape(3, 3, order="F")
            self.assertLess(np.linalg.norm(R.T @ R - np.eye(3)), 1e-4)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
