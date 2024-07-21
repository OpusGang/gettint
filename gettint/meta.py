from concurrent.futures import ThreadPoolExecutor
import vapoursynth as vs
import numpy as np
from typing import Dict, Callable, List, Tuple, Union
from functools import partial, wraps
from awsmfunc import fixlvls
core = vs.core

from metrics import Metrics

import traceback

class Adjustment:
    def __init__(self, name: str, func: Callable, **kwargs):
        self.name = name
        self.func = partial(func, **kwargs)

    def __call__(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self.func(clip)

class TestSuite:
    def __init__(
        self,
        prefilter: Callable = lambda clip: clip.resize.Bicubic(20, 20, filter_param_a=2, filter_param_b=0),
        metric: Metrics = Metrics.abs_difference,
        track_failures: bool = False,
        num_threads: int = vs.core.num_threads // 4
    ):

        self.adjustments: Dict[str, Adjustment] = {}

        self.frame: int = None
        self.prefilter: Callable | None = prefilter
        self.metric: Metrics = metric
        
        self.failed_tests: List[tuple[str, str]] = []
        self.track_failures: bool = track_failures
        self.num_threads: int = num_threads
        
        self.prefiltered_src: vs.VideoNode
        self.prefiltered_ref: vs.VideoNode

    def add_adjustment(self, name: str, func: Callable, **kwargs):
        self.adjustments[name] = Adjustment(name, func, **kwargs)

    def _apply_prefilter(self, src: vs.VideoNode, ref: vs.VideoNode):
        if self.prefilter:
            self.prefiltered_src = self.prefilter(src)
            self.prefiltered_ref = self.prefilter(ref)
        else:
            self.prefiltered_src = src
            self.prefiltered_ref = ref

    def _calculate_diff(self, adjustment: Adjustment) -> Tuple[str, float]:
        try:
            adjusted = adjustment(self.prefiltered_src)
            diff = self.metric(adjusted, self.prefiltered_ref)
            return adjustment.name, diff
        except Exception as e:
            error_message = f"Error in {adjustment.name}: {str(e)}\n{traceback.format_exc()}"

            if self.track_failures:
                self.failed_tests.append((adjustment.name, error_message))

            return adjustment.name, float('inf') if not Metrics.higher_is_better(self.metric) else float('-inf')

    def run_tests(self, src: vs.VideoNode, ref: vs.VideoNode, frame: int) -> Dict[str, float]:
        self.frame = frame
        self.failed_tests = []
        self._apply_prefilter(src, ref)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(self._calculate_diff, self.adjustments.values()))

        processed_results = dict(results)

        for name, diff in processed_results.items():
            print(f"{name}: {diff}")

        return processed_results

    def file_handler(self, src: str, ref: str) -> tuple[vs.VideoNode, vs.VideoNode]:
        clips = [
            core.bs.VideoSource(src),
            core.bs.VideoSource(ref)
        ]
        
        if self.frame is None:
            self.frame = clips[0].num_frames // 2
            
        clips = [clip[self.frame] for clip in clips]

        hcrop = 140 * 1080 // clips[0].width
        wcrop = 2 * hcrop
        
        for i in range(2):
            if clips[i].format.color_family == vs.YUV:
                clips[i] = clips[i].acrop.AutoCrop(
                    top=hcrop,
                    bottom=hcrop,
                    left=wcrop,
                    right=wcrop
                ).resize.Bicubic(format=vs.RGB24) # TODO fix hardcode RGB

        return clips[0], clips[1]

    def find_best_adjustment(self, src: str, ref: str, frame: int | None = None) -> tuple[str, float]:
        self.frame = frame
        src_clip, ref_clip = self.file_handler(src, ref)
        
        self._apply_prefilter(src_clip, ref_clip)

        # TODO: Implement least squares
        # self.add_least_squares_adjustment(self.prefiltered_src, self.prefiltered_ref)

        if self.prefiltered_src.width and self.prefiltered_src.height > 400 and self.num_threads > 6:
            raise Warning("You're probably about OOM, reduce thread count or resolution of your input")
            pass

        results = self.run_tests(self.prefiltered_src, self.prefiltered_ref, self.frame)

        if Metrics.higher_is_better(self.metric):
            best_adjustment = max(results, key=results.get)
        else:
            best_adjustment = min(results, key=results.get)
        
        print(f"\nBest adjustment: {best_adjustment} with difference: {results[best_adjustment]}")
        print(f"Tested {len(results)} combinations on frame {self.frame}")
        
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=Metrics.higher_is_better(self.metric))
        top_10_results = sorted_results[:10]

        print("\nTop 10 results:")
        for adjustment, value in top_10_results:
            print(f"{adjustment}: {value}")
        
        if self.track_failures and self.failed_tests:
            print("\nFailed tests:")
            for name, error_message in self.failed_tests:
                print(f"{name}:\n{error_message}\n")

        return best_adjustment, results[best_adjustment]

    def extended(self, base_name: str, func: Callable, operations: list[str]):
        for op_in in operations:
            for op_out in operations:
                if op_in != op_out:
                    self.add_adjustment(
                        f"{base_name} {op_in} to {op_out}",
                        func,
                        **{f"{base_name.lower()}_in": op_in, f"{base_name.lower()}_out": op_out}
                    )

    # TODO
    # Properly expand this method for all conceivable gamut problems.
    #
    # Currently, proper linearization is enforced; however, the possibility of improper
    # manipulation outside of linear space is entirely possible and will result in slight
    # color changes that don't match any other conversion.
    #
    # Similarly, an incorrect matrix conversion prior to this operation may also introduce
    # an otherwise unexpected tint that does not match conventional wisdom.
    # 
    # Consider a Y'CbCr signal with 709 matrices.
    # Convert this to RGB using an incorrect matrix (e.g., 601/470bg).
    # Linearize this using an incorrect transfer function (e.g., 709 -> linear instead of PQ -> linear).
    # Perhaps even use something like (709 -> 1886), or skip linearization entirely.
    # Then, proceed to manipulate the gamut within an assumed linear space.
    # -> Womp womp.
    # (Oh also they might somehow use entirely different values on the way out)
    # YUV 709 -> RGB 601 -> R'G'B (1886 -> Linear) -> R'G'B (Primaries) -> RGB (Linear -> SRGB) -> YUV 601; ect
    #
    # I'm thinking that an ideal complex method should handle Matrix, Primaries, and Transfers.
    #
    # Bonus points for handling white points. fmtconv lets us change W/R/G/B source-dest individually.
    #
    # fmtconv uses Bradford method
    # Libplacebo uses CAT97 or CAT16 https://github.com/haasn/libplacebo/commit/1fd3c7bde7b943fe8985c893310b5269a09b46c5
    # Others: VonKries, CAT02, HPE
    def complex(self, base_name: str, func: Callable, operations: Union[List[str], Dict[str, List[str]]]):
        if isinstance(operations, dict):
            primaries = operations.get('primaries', [])
            transfers = operations.get('transfers', [])

        for transfer in transfers:
            for prim_in in primaries:
                for prim_out in primaries:
                    if prim_in != prim_out:
                        self.add_adjustment(
                            f"{base_name} {prim_in} to {prim_out} (Transfer: {transfer})",
                            func,
                            primaries_in=prim_in,
                            primaries_out=prim_out,
                            transfer=transfer
                        )

class FormatAdjuster:
    @staticmethod
    def _adjust_format(clip: vs.VideoNode, target_format: vs.VideoFormat, matrix_s: str = None) -> vs.VideoNode:
        if clip.format != target_format:
            params = {
                "format": target_format,
                "filter_param_a": 1,
                "filter_param_b": 0
            }
            if matrix_s:
                params["matrix_s"] = matrix_s
            return clip.resize.Bicubic(**params)
        return clip

    @classmethod
    def yuv_adj(cls, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(clip: vs.VideoNode, *args: any, **kwargs: any) -> vs.VideoNode:
            adjusted_clip = cls._adjust_format(
                clip, 
                vs.YUV444P8, 
                matrix_s="709" if clip.format.color_family == vs.RGB else None
            )
            return func(adjusted_clip, *args, **kwargs)
        return wrapper

    @classmethod
    def gray_adj(cls, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(clip: vs.VideoNode, *args: any, **kwargs: any) -> vs.VideoNode:
            adjusted_clip = cls._adjust_format(
                clip, 
                vs.GRAY8, 
                matrix_s="709" if clip.format.color_family == vs.RGB else None
            )
            return func(adjusted_clip, *args, **kwargs)
        return wrapper

    @classmethod
    def rgb_adj(cls, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(clip: vs.VideoNode, *args: any, **kwargs: any) -> vs.VideoNode:
            adjusted_clip = cls._adjust_format(
                clip, 
                vs.RGB24, 
                matrix_s="709" if clip.format.color_family != vs.RGB else None
            )
            return func(adjusted_clip, *args, **kwargs)
        return wrapper

    @classmethod
    def rgbs_adj(cls, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(clip: vs.VideoNode, *args: any, **kwargs: any) -> vs.VideoNode:
            adjusted_clip = cls._adjust_format(
                clip, 
                vs.RGBS, 
                matrix_s="709" if clip.format.color_family != vs.RGB else None
            )
            return func(adjusted_clip, *args, **kwargs)
        return wrapper

class TintTests:
    def __init__(self):
        self.tests: Dict[str, Callable] = {}

    def add_test(self, name: str, func: Callable):
        self.tests[name] = func

    def run_tests(self, clip: vs.VideoNode) -> Dict[str, vs.VideoNode]:
        return {name: func(clip) for name, func in self.tests.items()}

    @staticmethod
    @FormatAdjuster.gray_adj
    def _gamma(c: vs.VideoNode, gamma: float) -> vs.VideoNode:
        return fixlvls(c, gamma=gamma)

    @staticmethod
    def _gamma_bug(c: vs.VideoNode) -> vs.VideoNode:
        return TintTests._gamma(c, 0.88)

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _TVtoPC(c: vs.VideoNode) -> vs.VideoNode:
        return c.resize.Point(range_in=0, range=1)

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _PCtoTV(c: vs.VideoNode) -> vs.VideoNode:
        return c.resize.Point(range_in=1, range=0)

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _0to16(c: vs.VideoNode) -> vs.VideoNode:
        return fixlvls(c, 1, min_in=[0, 0])

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _255to235(c: vs.VideoNode) -> vs.VideoNode:
        return fixlvls(c, 1, max_in=[255, 255])

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _601to709(c: vs.VideoNode) -> vs.VideoNode:
        return c.resize.Point(matrix_in_s='470bg', matrix_s='709')

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _709to601(c: vs.VideoNode) -> vs.VideoNode:
        return c.resize.Point(matrix_s='470bg', matrix_in_s='709')

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _truncate(c: vs.VideoNode) -> vs.VideoNode:
        c = core.resize.Bicubic(c, format=vs.YUV420P10)
        c = c.std.Expr("x 2 +")
        return core.resize.Bicubic(c, format=vs.YUV444P8)

    @staticmethod
    @FormatAdjuster.yuv_adj
    def _ceil(c: vs.VideoNode) -> vs.VideoNode:
        c = core.resize.Bicubic(c, format=vs.YUV420P10)
        c = c.std.Expr("x 2 -")
        return core.resize.Bicubic(c, format=vs.YUV444P8)

    @staticmethod
    @FormatAdjuster.rgb_adj
    def _offset_gain(c: vs.VideoNode, gains: list[float], offsets: list[float]) -> vs.VideoNode:
        return c.std.Expr([f"x {gains[i]} * {offsets[i]} +" for i in range(c.format.num_planes)])

    @staticmethod
    @FormatAdjuster.rgb_adj
    def _gamma_rgb(c: vs.VideoNode, gammas: list[float]) -> vs.VideoNode:
        max_val = (2 << c.format.bits_per_sample) - 1
        return c.std.Expr([f"x {max_val} / {gammas[i]} pow {max_val} *" for i in range(c.format.num_planes)])

    @staticmethod
    @FormatAdjuster.rgb_adj
    def Transfer(c: vs.VideoNode, transfer_in: str, transfer_out: str) -> vs.VideoNode:
        return c.fmtc.transfer(transs=transfer_in, transd=transfer_out)
    
    @staticmethod
    @FormatAdjuster.yuv_adj
    def Matrix(c: vs.VideoNode, matrix_in: str, matrix_out: str) -> vs.VideoNode:
        return core.fmtc.matrix(c, mats=matrix_in, matd=matrix_out)

    @staticmethod
    @FormatAdjuster.rgb_adj
    def Primaries(c: vs.VideoNode, primaries_in: str, primaries_out: str, transfer: str) -> vs.VideoNode:
        linear = c.fmtc.transfer(transs=transfer, transd="linear")
        prims = linear.fmtc.primaries(prims=primaries_in, primd=primaries_out)
        return prims.fmtc.transfer(transs="linear", transd=transfer)
