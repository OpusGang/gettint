import vapoursynth as vs
core = vs.core
from vsutil import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections.abc import Callable
from scipy import optimize

def matchcolors(src: vs.VideoNode,
                ref: vs.VideoNode,
                params: str | list[str],
                frame: int | None = None,
                fmt: vs.PresetFormat = vs.RGB24,
                placebo: bool = False,
                plot_points: bool = False,
                plotstep: int = 3,
                denoise: Callable = lambda clip: core.dfttest.DFTTest(clip, tbsize=1),
                matrix: str | None = None,
                fname: str | None = "fig.pdf",
                tv_range: bool = True,
                return_std: bool = False,
                silence: bool = False) -> list:
    """
    A simple script to match pixel values from src clip to ref clip, perform 
    lsq regression on the ref->src mappings and print params.
    This can be used to determine simple color corrections.

    :param src: To-be-detinted video.  Must be 4:4:4 if YUV fmt.
    :param ref: Reference video.  Must be 4:4:4 if YUV fmt.
    :param params: Parameters to test for.  Can be a string or list.
                   It is recommended to test for as few params as possible.
                   Possible params:
                    - gamma: exponentiation (values mapped to [0, 1])
                    - gain: multiplication
                    - offset: addition 
                    - levels: map between ranges, can't be combined
    :param frame: Frame number to test.  Defaults to middle.
    :param fmt: Format to test parameters in.
    :param placebo: Use statistics module instead of numpy for mean, std dev.
    :param plot_points: Plot mean every plotstep points.
    :param plotstep: See plot_points.
    :param denoise: Denoise function.  Defaults to spatial DFTTest.
    :param matrix: Conversion matrix if necessary.  Default is BT.709.
    :param fname: Output figure filename.  None/False disables.
    :param tv_range: TV/PC range switch. Default is TV.  No effect for RGB.
    :param return_std: Whether to return standard deviation along with params.
    :param silence: Suppress output.
    """
    
    if "YUV" in fmt.name:
        if "444" not in fmt.name:
            raise ValueError("Currently, only 4:4:4 is supported.")
    if matrix is None and any(f in fmt.name for f in ["GRAY", "YUV"]):
        matrix = "709"

    clips = [ref, src]

    if frame is None:
        frame = round(len(clips[0]) / 2)
   
    # extract values from clips
    values = []
    for f in range(2):
        # a quick denoise
        if denoise:
            clips[f] = denoise(clips[f])
        # determine format
        if clips[f].format != fmt:
            clips[f] = clips[f].resize.Bicubic(format=fmt, matrix_s=matrix, filter_param_a=0.5, filter_param_b=0)
        # pel values into np array
        values.append(np.array(clips[f].get_frame(frame)))
      
    # labels and colors according to format
    if fmt.name.startswith("RGB"):
        pnames = "RGB"
        colors = ["red", "green", "blue"]
        colors2 = ["darkred", "darkgreen", "darkblue"]
        # this is useful for fitting later
        max_out = [scale_value(255, 8, clips[0].format.bits_per_sample)]
        min_out = [scale_value(0, 8, clips[0].format.bits_per_sample)]
    elif fmt.name.startswith("YUV"):
        pnames = "YUV"
        colors = ["dimgray", "blue", "green"]
        colors2 = ["black", "darkblue", "darkgreen"]
        # have to check for range with YUV
        if tv_range:
            max_out = [scale_value(235, 8, clips[0].format.bits_per_sample),
                    scale_value(240, 8, clips[0].format.bits_per_sample, chroma=True)]
            min_out = [scale_value(16, 8, clips[0].format.bits_per_sample),
                    scale_value(16, 8, clips[0].format.bits_per_sample, chroma=True)]
        else:
            max_out = [scale_value(255, 8, clips[0].format.bits_per_sample),
                    scale_value(255, 8, clips[0].format.bits_per_sample, chroma=True)]
            min_out = [scale_value(0, 8, clips[0].format.bits_per_sample),
                    scale_value(0, 8, clips[0].format.bits_per_sample, chroma=True)]
    elif fmt.name.startswith("GRAY"):
        pnames = "G"
        colors = ["dimgray"]
        colors2 = ["black"]
        # also have to check range for GRAY
        if tv_range:
            max_out = [scale_value(235, 8, clips[0].format.bits_per_sample)]
            min_out = [scale_value(16, 8, clips[0].format.bits_per_sample)]
        else:
            max_out = [scale_value(255, 8, clips[0].format.bits_per_sample)]
            min_out = [scale_value(0, 8, clips[0].format.bits_per_sample)]
    else:
        raise ValueError("Unknown format!")
     
    guess_dict = {
            "gamma"  : [1, 1],
            "gain"   : [1, 1],
            "offset" : [0, 0],
            "levels" : [[min_out[i], max_out[i]] for i in range(len(min_out))]
            }

    param_names_dict = {
            "gamma"  : "gamma",
            "gain"   : "gain",
            "offset" : "offset",
            "levels" : ["min_in", "max_in"]
            }

    if isinstance(params, str):
        params = [params]
    # levels along with others is too much of a pita rn
    if "levels" in params:
        if len(params) > 1:
            raise ValueError("Levels is only supported on its own")
        func = [lambda x, min_in, max_in: ((x - min_in) / (max_in - min_in)) * (max_out[i] - min_out[i]) + min_out[i]
                for i in range(len(min_out))]
    else:
        # generate a string with the function we want to fit to
        def _strip_param(func, param, value):
            return func.replace(", " + param, "").replace(param, str(value))

        func = "[lambda x, gamma, gain, offset: (((x - min_out[i]) / (max_out[i] - min_out[i])) ** gamma * (max_out[i] - min_out[i]) + min_out[i]) * gain + offset \
                for i in range(len(min_out))]"

        for param in guess_dict.keys():
            if param not in params and param != "levels":
                value = guess_dict[param]
                if isinstance(value, list):
                    value = str(value) + "[i]"
                func = _strip_param(func, param, value)
        func = eval(func, {"min_out": min_out, "max_out": max_out})

    param_names = []
    guess = []
    for p in params:
        par = param_names_dict[p]
        if isinstance(par, list):
            param_names += par
        else:
            param_names.append(par)
    for i in range(min(clips[0].format.num_planes, 2)):
        guess.append([])
        for p in params:
            gue = guess_dict[p]
            guess[i].append(gue[i])

    if placebo:
        import statistics
        mean = statistics.mean
        std = lambda v: max(statistics.stdev(v), 1) if len(v) > 1 else 1
    else:
        mean = np.mean
        std = lambda v: max(np.std(v), 1)

    fit_params = []
    # create dicts of matching values
    for i in range(clips[0].format.num_planes):
        x = list(itertools.chain.from_iterable(values[0][i]))
        y = list(itertools.chain.from_iterable(values[1][i]))
        mappings = {}
        for j in range(len(x)):
            try:
                mappings[x[j]] += [y[j]]
            except:
                mappings[x[j]] = [y[j]]
        # x, y, std devs
        xvals = np.array(list(mappings.keys()))
        yvals = [mean(vals) for vals in mappings.values()]
        sigma = [std(vals) for vals in mappings.values()]
        
        if plot_points:
            plt.plot(xvals[i::plotstep], yvals[i::plotstep], 'x', label=pnames[i], color=colors[i])

        # grab right func/guess
        current_func = func[min(i, len(func) - 1)]
        current_guess = guess[min(i, len(guess) - 1)]

        # the actual fitting
        popt, pcov = optimize.curve_fit(current_func, xvals, yvals, p0=current_guess, sigma=sigma, maxfev=5000)
        if return_std:
            fit_params.append([popt, np.sqrt(np.diag(pcov))])
        else:
            fit_params.append(popt)
        if not silence:
            for j in range(len(popt)):
                print(f"{pnames[i]} {param_names[j]}: {round(popt[j], 3)}")
        plt.plot(xvals, current_func(xvals, *popt), label=pnames[i] + " fit", color=colors2[i])
    
    # plot it
    if fname:
        plt.xlabel("Reference")
        plt.ylabel("Source")
        plt.legend()
        plt.grid()
        plt.savefig(fname)

    return fit_params

