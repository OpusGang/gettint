from functools import lru_cache
import vapoursynth as vs
core = vs.core
import numpy as np

class Metrics:
    @staticmethod
    @lru_cache(maxsize=None)
    def _get_frame(clip: vs.VideoNode, n: int = 0) -> vs.VideoFrame:
        return clip.get_frame(n)

    @staticmethod
    @lru_cache(maxsize=None)
    def _to_array(frame: vs.VideoFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, vs.VideoFrame):
            return np.array(frame)
        return frame

    # TODO
    # BROKEN YUV INPUT
    @staticmethod
    def CIE2000(ref: vs.VideoNode | np.ndarray, adjusted: vs.VideoNode | np.ndarray) -> float:
        if isinstance(ref, vs.VideoNode):
            ref = ref.resize.Point(format=vs.YUV420P12, matrix_s="709")
            adjusted = adjusted.resize.Point(format=vs.YUV420P12, matrix_s="709")
            metric = core.vmaf.Metric(ref, adjusted, feature=4)
            return Metrics._get_frame(metric).props['ciede2000']
        else:
            raise ValueError("CIE2000 metric requires VideoNode inputs")

    @staticmethod
    def abs_difference(ref: vs.VideoNode, adjusted: vs.VideoNode) -> float:
        # TODO
        # wasted cycles
        ref = ref.resize.Bicubic(format=vs.RGBS)
        adjusted = adjusted.resize.Bicubic(format=vs.RGBS)

        diff = core.std.Expr([ref, adjusted], "x y - abs")
        diff_frame = Metrics._get_frame(diff)
        diff_array = Metrics._to_array(diff_frame)
        return float(np.mean(diff_array))

    @staticmethod
    def planestats(ref: vs.VideoNode | np.ndarray, adjusted: vs.VideoNode | np.ndarray) -> float:
        if isinstance(ref, vs.VideoNode):
            ref = ref.resize.Bicubic(format=vs.RGB24)
            adjusted = adjusted.resize.Bicubic(format=vs.RGB24)
            stats = core.std.PlaneStats(ref, adjusted)
            return Metrics._get_frame(stats).props['PlaneStatsDiff']
        else:
            raise ValueError("Planestats metric requires VideoNode inputs")

    @staticmethod
    def higher_is_better(metric) -> bool:
        return metric in [Metrics.CIE2000]
