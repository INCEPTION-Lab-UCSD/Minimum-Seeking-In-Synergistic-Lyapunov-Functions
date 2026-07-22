# Minimum Seeking in Synergistic Lyapunov Functions

## MuJoCo SO(3) animation

Render the default SO(3) attitude-stabilization scenario from the repository root:

```bash
uv run python src/so3_mujoco.py
```

The video is written to `Animations/so3_mujoco.mp4`. Resolution, frame rate,
frame count, and output format are configurable:

```bash
uv run python src/so3_mujoco.py \
  --output Animations/so3_mujoco.gif \
  --fps 30 --frames 300 --width 960 --height 720
```

An existing `SO3` simulation and solution can also be rendered directly:

```python
simulation.render_mujoco(solution, "Animations/my_so3_run.mp4", fps=30)
```
