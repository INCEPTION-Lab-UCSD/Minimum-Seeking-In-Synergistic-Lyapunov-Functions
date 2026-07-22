import math

import matplotlib.pyplot as plt
import numpy as np

import hybrid_solution
import obstacle_avoidance
import so3
import sphere_2d
import sphere_3d


def run_sphere_2d():
    p0 = np.array([0, -1])
    eta0 = np.array([1, 0])
    q0 = 1

    gamma = np.sqrt(1)
    kappa = 4
    epsilon = 1 / np.sqrt(4 * np.pi)
    delta = 0.25
    t_1 = 0.0
    t_2 = 25.0
    p_target = np.array([0, 1])
    simulation = sphere_2d.Sphere_2D(
        p0, q0, p_target, gamma, kappa, epsilon, t_1, t_2, delta=delta, eta0=eta0
    )
    solution = simulation.solve()

    _, animation = simulation.animate(solution)

    plt.show()
    return animation


def run_obstacle_avoidance():
    z0 = np.array([0, -4], dtype=float)
    target = np.array([0, 2], dtype=float)
    q0 = 1
    eta0 = np.array([1, 0], dtype=float)
    obstacle_radius = 1
    obstacle_center = np.array([0, 0], dtype=float)
    epsilon = 1 / np.sqrt(6 * np.pi)
    gamma = 2.0
    chi_1 = 1.0
    chi_2 = 0.5
    t_1 = 0.0
    t_2 = 40.0
    delta = 0.25
    kappa = 4.0
    T_0 = 1.0

    simulation = obstacle_avoidance.Target_Seeking(
        z0,
        q0,
        eta0,
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
        obstacle_center=obstacle_center,
        theta_seed=0,
    )

    solution = simulation.solve()

    _, animation = simulation.animate_obstacle_avoidance(solution)

    plt.show()
    return animation


def run_sphere_3d(show=True):
    control_gains_constants = np.array([3.0, 2.0, 1.0])
    target = np.array([0, 0, 1], dtype=float)
    gamma = 1.0
    kappa = 4.0
    delta = 0.2
    epsilon = 1 / np.sqrt(8 * np.pi)

    p0 = np.array([0, 0, -1], dtype=float)
    eta0 = np.tile(np.array([1.0, 0.0]), (3, 1))
    q0 = 1
    t_1 = 0.0
    t_2 = 15.0
    chi_1 = 1.0
    chi_2 = 0.5

    simulation = sphere_3d.Sphere_3D(
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
        control_gains_constants,
        theta_seed=0,
    )

    solution = simulation.solve()
    _, animation = simulation.animate(solution, frame_count=1000)
    if show:
        plt.show()
    return animation


def run_so3(show=True):
    p0 = np.diag([-1.0, 1.0, -1.0]).reshape(-1, order="F")
    eta0 = np.tile(np.array([1.0, 0.0]), (3, 1))
    q0 = 1
    gamma = 1
    kappa = 4
    angular_velocity = np.array([11, 12, 13], dtype=float)
    control_gain_constants = np.array([1, 2, 3], dtype=float)
    delta = 0.2
    epsilon = 1 / np.sqrt(12 * np.pi)
    target = np.eye(3).reshape(-1, order="F")
    t_1 = 0.0
    t_2 = 10.0
    chi_1 = 1.0
    chi_2 = 0.5

    simulation = so3.SO3(
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
        control_gain_constants=control_gain_constants,
        angular_velocity=angular_velocity,
        theta_seed=0,
    )

    solution = simulation.solve()
    _, animation = simulation.animate(solution, frame_count=1000)
    if show:
        plt.show()
    return animation


if __name__ == "__main__":
    run_sphere_2d()
    # run_obstacle_avoidance()
    # sphere_animation = run_sphere_3d(show=False)
    # so3_animation = run_so3(show=False)
    plt.show()
