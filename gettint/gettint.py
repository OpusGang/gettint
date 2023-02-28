from .matchcolors import *
import awsmfunc as awf

# shitty decorators
def yuv_adj(func):
    def inner(*args):
        c = args[0]
        if c.format != vs.YUV444P8:
            if c.format.color_family == vs.RGB:
                c = c.resize.Bicubic(format=vs.YUV444P8, matrix_s="709", filter_param_a=0.5, filter_param_b=0)
            else:
                c = c.resize.Bicubic(format=vs.YUV444P8, filter_param_a=0.5, filter_param_b=0)
        return func(c, *args[1:])
    return inner


def gray_adj(func):
    def inner(*args):
        c = args[0]
        if c.format != vs.GRAY8:
            if c.format.color_family == vs.RGB:
                c = c.resize.Bicubic(format=vs.GRAY8, matrix_s="709", filter_param_a=0.5, filter_param_b=0)
            else:
                c = c.resize.Bicubic(format=vs.GRAY8, filter_param_a=0.5, filter_param_b=0)
        return func(c, *args[1:])
    return inner


def rgb_adj(func):
    def inner(*args):
        c = args[0]
        if c.format != vs.RGB24:
            if c.format.color_family == vs.RGB:
                c = c.resize.Bicubic(format=vs.RGB24, filter_param_a=0.5, filter_param_b=0)
            else:
                c = c.resize.Bicubic(format=vs.RGB24, matrix_s="709", filter_param_a=0.5, filter_param_b=0)
        return func(c, *args[1:])
    return inner


# common tints
@gray_adj
def _gamma(c, gamma):
    return awf.fixlvls(c, gamma=gamma)


def _gamma_bug(c):
    return _gamma(c, 0.88)

@yuv_adj
def _TVtoPC(c):
    return c.resize.Point(range_in=0, range=1)


@yuv_adj
def _PCtoTV(c):
    return c.resize.Point(range_in=1, range=0)


@yuv_adj
def _0to16(c):
    return awf.fixlvls(c, 1, min_in=[0, 0])


@yuv_adj
def _255to235(c):
    return awf.fixlvls(c, 1, max_in=[255, 255])


@yuv_adj
def _601to709(c):
    return c.resize.Point(matrix_in_s='470bg', matrix_s='709')


@yuv_adj
def _709to601(c):
    return c.resize.Point(matrix_s='470bg', matrix_in_s='709')


@yuv_adj
def _truncate(c):
    c = depth(c, 10)
    c = c.std.Expr("x 2 +")
    return depth(c, 8)


@yuv_adj
def _ceil(c):
    c = depth(c, 10)
    c = c.std.Expr("x 2 -")
    return depth(c, 8)


@rgb_adj
def _offset_gain(c, gains, offsets):
    return c.std.Expr([f"x {gains[i]} * {offsets[i]} +" for i in range(c.format.num_planes)])


@rgb_adj
def _gamma_rgb(c, gammas):
    max_val = (2 << c.format.bits_per_sample) - 1
    return c.std.Expr([f"x {max_val} / {gammas[i][0]} pow {max_val} *" for i in range(c.format.num_planes)])


@yuv_adj
def _levels(c, levels):
    c = split(c)
    maxes = [235, 240]
    for i in range(len(c)):
        c[i] = c[i].std.Levels(min_in=levels[min(1, i)][0], max_in=levels[min(1, i)][1],
                min_out=16, max_out=maxes[min(1, i)])
    return join(c)


def gettint(src: str,
            ref: str,
            frame: int = None,
            force_matchcolors: bool = False):
    """
    A script to automate finding the right detinting method.

    :param src: Tinted source.
    :param ref: Untinted reference.
    :param frame: Frame number.  Defaults to middle of clip.
    :param force_matchcolors: Force matchcolors script by bypassing common checks.
    """
    clips = [core.lsmas.LWLibavSource(src), core.lsmas.LWLibavSource(ref)]

    hcrop = 140 * 1080 / clips[0].width
    wcrop = 2 * hcrop
    for i in range(2):
        if clips[i].format.color_family == vs.YUV:
            clips[i] = clips[i].acrop.AutoCrop(top=hcrop, bottom=hcrop, left=wcrop, right=wcrop)
        clips[i] = clips[i].dfttest.DFTTest(tbsize=1)
    
    if frame is None:
        frame = round(len(clips[0]) / 2)

    def _test_adj(adj, a, b):
        a = adj(a)
        try:
            b = b.resize.Bicubic(format=a.format, filter_param_a=0.5, filter_param_b=0)
        except vs.Error:
            b = b.resize.Bicubic(format=a.format, matrix_s="709", filter_param_a=0.5, filter_param_b=0)
        diff = core.std.Expr([a, b], "x y - abs")
        diff = np.array(diff.get_frame(frame))
        return np.mean(diff)

    # adjustment function, threshold under which this can be assumed to be correct
    common_tests = {
            "gamma bug"   : [_gamma_bug, 5],
            "TV->PC"      : [_TVtoPC, 3],
            "PC->TV"      : [_PCtoTV, 3],
            "0->16"       : [_0to16, 2],
            "255->235"    : [_255to235, 1.5],
            "601->709"    : [_601to709, 0.5],
            "709->601"    : [_709to601, 0.5],
            "truncate"    : [_truncate, 1],
            "ceil"    : [_ceil, 1]
            }

    backup_tests = {
            "gamma"       : lambda c: _gamma(c, match_res[0][0]),
            "gamma RGB"   : lambda c: _gamma_rgb(c, match_res[1]),
            "offset+gain" : lambda c: _offset_gain(c, [round(vals[0], 3) for vals in match_res[2]], [round(vals[1], 3)
                for vals in match_res[2]]),
            "levels"      : lambda c: _levels(c, match_res[3])
            }

    if not force_matchcolors:
        # try to find the tint in common_tests
        results = {}
        for test in common_tests.keys():
            results[test] = _test_adj(common_tests[test][0], clips[0], clips[1])
            print(f"{test} difference: {round(results[test], 3)}")
    
        # find lowest diff and check if below predefined threshold
        lowest = min(results.values())
        lowest_key = [key for key in results if results[key] == lowest and results[key] < common_tests[key][1]]
        if lowest_key:
            print("Likeliest tint fix(es):", ", ".join(lowest_key))
            return
    
        print("No common tint identified.  Running backup matchcolors tests.")
    else:
        results = {}

    # fall back to running matchcolors and try backup_tests
    match_tests = ["gamma", "gamma", ["gain", "offset"], "levels"]
    match_fmts = [vs.YUV444P8, vs.RGB24, vs.RGB24, vs.YUV444P8]
    match_res = []
    
    for i in range(len(match_tests)):
        try:
            match_res.append(matchcolors(clips[0], clips[1], match_tests[i], fmt=match_fmts[i], fname=False, denoise=False))
        # handling grayscale clips in yuv
        except TypeError as e:
            if match_fmts[i] == vs.YUV444P8:
                match_res.append(matchcolors(clips[0], clips[1], match_tests[i], fmt=vs.GRAY8, fname=False, denoise=False))
            else:
                raise e

    for test in backup_tests.keys():
        results[test] = _test_adj(backup_tests[test], clips[0], clips[1])
        print(f"{test} difference: {round(results[test], 3)}")

    # no need to check if under threshold anymore
    lowest = min(results.values())
    lowest_key = [key for key in results if results[key] == lowest]
    print("Likeliest tint fix(es):", ", ".join(lowest_key))
    return


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find the likeliest tint to match one source to another")
    parser.add_argument(dest="src", type=str, help="Path to source file")
    parser.add_argument(dest="ref", type=str, help="Path to reference file")
    parser.add_argument("-f", "--frame", dest="frame", type=int, help="Frame number", required=False, default=None)
    parser.add_argument("-m", "--force-matchcolors", dest="force_matchcolors", help="Force matchcolors", action="store_true")

    args = parser.parse_args()

    gettint(args.src, args.ref, args.frame, args.force_matchcolors)

if __name__ == "__main__":
    main()

