from meta import Metrics, TestSuite, TintTests
import vapoursynth as vs
core = vs.core

# TODO
# Do something about these lists

matrices_zimg = ["709", "fcc", "470bg", "240m", "ycgco"]
matrices_fmtc = ["601", "709", "2020", "240", "FCC", "YCoCg", "YDzDx", "RGB"]
matrices_full = matrices_zimg + matrices_fmtc

transfers = [
    "709", "470m", "240", "linear", "pq",
    "428", "hlg", "1886", "sigmoid", "srgb"
    ]

transfers_full = [
    "709", "470m", "470bg", "240", "linear",
    "log100", "log316", "61966-2-4", "1361",
    "srgb", "pq", "428", "hlg", "1886", "1886a",
    "filmstream", "slog", "slog2", "slog3", "logc2",
    "logc3", "canonlog", "adobergb", "romm", "acescc", 
    "acescct", "erimm", "vlog", "davinci", "log3g10",
    "redlog", "cineon", "panalog", "sigmoid"
    ]

primaries = ["709", "ntsc", "ntscj", "pal", "240m", "filmc", "2020"]
primaries_full = [
    "709", "ntsc", "ntscj", "pal", "240m", "filmc", "2020",
    "scrgb", "adobe98", "adobewide", "apple", "romm",
    "ciergb", "ciexyz", "p3dci", "p3d65", "p3d60",
    "p3p", "cinegam", "3213", "aces", "ap1", "sgamut",
    "sgamut3cine", "alexa", "vgamut", "p22", "fs",
    "davinci", "dragon", "dragon2", "red", "red2", "red3", "red4", "redwide"
    ]


suite = TestSuite(
    prefilter=lambda clip: clip.resize.Bicubic(20, 20, filter_param_a=2, filter_param_b=0),
    metric=Metrics.abs_difference, track_failures=True
    )

suite.add_adjustment("Gamma Bug", TintTests._gamma_bug)
suite.add_adjustment("TV to PC", TintTests._TVtoPC)
suite.add_adjustment("PC to TV", TintTests._PCtoTV)
suite.add_adjustment("0 to 16", TintTests._0to16)
suite.add_adjustment("255 to 235", TintTests._255to235)
suite.add_adjustment("601 to 709", TintTests._601to709)
suite.add_adjustment("709 to 601", TintTests._709to601)
suite.add_adjustment("Truncate", TintTests._truncate)
suite.add_adjustment("Ceil", TintTests._ceil)
suite.add_adjustment("Gamma 1.2", TintTests._gamma, gamma=1.2)
suite.add_adjustment("Offset+Gain", TintTests._offset_gain, gains=[1.1, 1.0, 1.0], offsets=[5, 0, 0])


suite.extended("Transfer", TintTests.Transfer, transfers_full)
suite.complex(
    "Primaries",
    TintTests.Primaries,
    {
        'primaries': primaries,
        'transfers': transfers
    }
)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find the likeliest tint to match one source to another")
    parser.add_argument(dest="src", type=str, help="Path to source file")
    parser.add_argument(dest="ref", type=str, help="Path to reference file")
    parser.add_argument("-f", "--frame", dest="frame", type=int, help="Frame number", required=False, default=None)
    parser.add_argument("-m", "--force-matchcolors", dest="force_matchcolors", help="Force matchcolors", action="store_true")
    parser.add_argument("-c", "--checks", dest="mode", help="Checks to perform. Either standard, extended, or full", type=str, default="standard")

    args = parser.parse_args()

    suite.find_best_adjustment(args.src, args.ref, args.frame)

if __name__ == "__main__":
    main()
