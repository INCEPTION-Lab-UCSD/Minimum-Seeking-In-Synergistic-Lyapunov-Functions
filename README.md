# Minimum-Seeking in Synergistic Lyapunov Functions

Python simulations and visualizations for examples from [“Hybrid minimum-seeking in synergistic Lyapunov functions: Robust global stabilization under unknown control directions”](https://doi.org/10.1016/j.automatica.2026.113112).

## Setup

Install the project dependencies:

```bash
uv sync
```

Run commands from the repository root. The examples use Matplotlib for plots
and animations, so a Python environment with GUI support is required for
interactive windows.

## Running simulations and viewing plots

The simulation entry points are defined in `src/main.py`. Run the default
SO(3) example:

```bash
uv run python src/main.py
```

By default, this runs `run_so3()` and opens the Matplotlib animation with
`plt.show()`. Close the plot window to return to the terminal.

To run a different example, edit the `if __name__ == "__main__":` block in
`src/main.py` and uncomment the function you want to run:

```python
if __name__ == "__main__":
    # run_sphere_2d()
    # run_obstacle_avoidance()
    run_sphere_3d()
    # run_so3()
    # run_nonholonomic()
```

Available examples:

- `run_sphere_2d()` simulates minimum-seeking on the unit circle and opens a
  2D animation.
- `run_obstacle_avoidance()` simulates target seeking around an obstacle and
  opens an obstacle-avoidance animation.
- `run_sphere_3d()` simulates minimum-seeking on the unit sphere and opens a
  3D animation.
- `run_so3()` simulates attitude stabilization on `SO(3)` and opens a 3D
  animation.
- `run_nonholonomic()` simulates a nonholonomic vehicle with obstacle
  avoidance and opens an animation.

Pass `save=True` to an example function to write its animation to the
repository’s `Animations/` directory. The default filenames are
`circle.gif`, `obstacle_avoidance.gif`, `sphere_3d.mp4`, `so3.mp4`, and
`nonholonomic.gif`, respectively.

The simulation classes also expose plotting helpers such as `plot()`,
`plot_obstacle_avoidance()`, and `plot_trajectories_and_control_gains()`.
These can be called from a Python session after solving a simulation, or used
by replacing the corresponding `animate(...)` call in `src/main.py`.

## MuJoCo SO(3) visualization

The SO(3) kinematics and hybrid solution are computed by `src/so3.py`.
MuJoCo is used as a visualization and playback layer for the solved attitude
trajectory. Render the default scenario from the repository root:

```bash
uv run python src/so3_mujoco.py
```

The video is written to `Animations/so3_mujoco.mp4`. Configure the output,
frame rate, frame count, and resolution with command-line options:

```bash
uv run python src/so3_mujoco.py \
  --output Animations/so3_mujoco.gif \
  --fps 30 --frames 300 --width 960 --height 720
```

An existing `SO3` simulation and solution can also be rendered directly:

```python
simulation.render_mujoco(solution, "Animations/my_so3_run.mp4", fps=30)
```

## Citation

- M. Abdelgalil and J. I. Poveda, [**Hybrid minimum-seeking in synergistic Lyapunov functions: Robust global stabilization under unknown control directions**](https://doi.org/10.1016/j.automatica.2026.113112), *Automatica*, vol. 191, article 113112, 2026.

```bibtex
@article{abdelgalil2026hybrid,
  title={Hybrid minimum-seeking in synergistic Lyapunov functions: Robust global stabilization under unknown control directions},
  author={Abdelgalil, Mahmoud and Poveda, Jorge I.},
  journal={Automatica},
  volume={191},
  pages={113112},
  year={2026},
  doi={https://doi.org/10.1016/j.automatica.2026.113112}
}
```

## License

This project is distributed under the [MIT License](LICENSE).
