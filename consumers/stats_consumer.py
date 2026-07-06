import numpy as np


class StatsConsumer:
    def __init__(self, n_frames):
        self.frame_means = np.empty(n_frames, dtype=np.float64)
        self.px_max, self.total, self.count_px = 0.0, 0.0, 0.0

    def update(self, i, frame, *args):
        self.frame_means[i] = frame.mean()
        self.total += frame.sum()
        self.count_px += frame.size
        self.px_max = max(self.px_max, float(frame.max()))

    def finalize(self):
        pass

    def results(self, n_frames):
        return {
            "frame_count": n_frames,
            "max_brightness": self.px_max,
            "mean_brightness": self.total / self.count_px,
            "brightness_std": float(self.frame_means.std()),
            "motion": float(np.abs(np.diff(self.frame_means)).mean()),
        }
