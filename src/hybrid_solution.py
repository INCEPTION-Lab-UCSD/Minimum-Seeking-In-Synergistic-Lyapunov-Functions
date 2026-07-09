import numpy as np


class HybridSolution:
    """
    Wraps the sequence of per-segment scipy.integrate solve_ivp results
    produced by VehicleTrajectorySimultion.solve() into a single piecewise
    trajectory, so the hybrid solution can be queried like a normal
    solve_ivp result (callable dense output, plus concatenated .t / .y).
    """

    def __init__(self, segments):
        self.segments = segments
        self.t = np.concatenate([seg.t for seg in segments])
        self.y = np.concatenate([seg.y for seg in segments], axis=1)
        self.switch_times = [seg.t[0] for seg in segments[1:]]

    def __call__(self, t):
        """Evaluate the stitched trajectory at scalar or array time t."""
        t_arr = np.atleast_1d(t).astype(float)
        out = np.empty((self.segments[0].y.shape[0], len(t_arr)))
        for i, ti in enumerate(t_arr):
            seg = self._segment_for_time(ti)
            out[:, i] = seg.sol(ti)
        return out[:, 0] if np.isscalar(t) else out

    def _segment_for_time(self, t):
        for seg in self.segments:
            if seg.t[0] - 1e-12 <= t <= seg.t[-1] + 1e-12:
                return seg
        # clamp out-of-range queries to the nearest boundary segment
        return self.segments[0] if t < self.segments[0].t[0] else self.segments[-1]

    def mode_at(self, t):
        """Return the active z_1 mode (rounded to int) at time t."""
        y = self(t)
        if y.ndim == 1:
            return int(round(y[5]))
        return np.round(y[5]).astype(int)
