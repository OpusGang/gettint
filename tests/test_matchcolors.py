from gettint.matchcolors import matchcolors
import vapoursynth as vs
core = vs.core
import unittest

class TestMethods(unittest.TestCase):

    def setUp(self):
        self.img = core.imwri.Read("test_image.png")
        self.vid = self.img.resize.Spline36(format=vs.YUV444P8, matrix=1) * 3 # we don't actually use any temporal info anyway

    def test_rgb_gain(self):
        print("Testing RGB gain")
        gain = 0.5
        ref = self.img.std.Expr(f"x {gain} *")
        expect = [[1 / gain]] * 3
        res = matchcolors(self.img, ref, params="gain", fname=False, denoise=False)
        for i in range(len(expect)):
            for j in range(len(expect[i])):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 0.5)
        return

    def test_rgb_gain_offset(self):
        print("Testing RGB gain and offset")
        gain = 0.5
        offset = 0.5
        ref = self.img.std.Expr(f"x {gain} * {offset} +")
        expect = [[1 / gain, -offset]] * 3
        res = matchcolors(self.img, ref, params=["gain", "offset"], fname=False, denoise=False)
        for i in range(len(expect)):
            for j in range(len(expect[i])):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 0.5)
        return

    def test_rgb_gamma(self):
        print("Testing RGB gamma")
        gamma = 0.5
        ref = self.img.std.Levels(gamma=gamma)
        expect = [[gamma]] * 3
        res = matchcolors(self.img, ref, params="gamma", fname=False, denoise=False)
        for i in range(len(expect)):
            for j in range(len(expect[i])):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 0.25)
        return

    def test_rgb_offset(self):
        print("Testing RGB offset")
        offset = 0.5
        ref = self.img.std.Expr(f"x {offset} +")
        expect = [[-offset]] * 3
        res = matchcolors(self.img, ref, params="offset", fname=False, denoise=False)
        for i in range(len(expect)):
            for j in range(len(expect[i])):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 0.5)
        return

    def test_yuv_levels(self):
        print("Testing YUV levels")
        img = self.img.resize.Spline36(format=vs.YUV444P8, matrix_s="709")
        ref = img.std.Levels(min_in=16, max_in=235, min_out=30, max_out=215, planes=0)
        expect = [[30, 215], [16, 240], [16, 240]]
        res = matchcolors(img, ref, fmt=vs.YUV444P8, params="levels", fname=False, denoise=False)
        for i in range(3):
            for j in range(2):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 10)
        return

    def test_yuv_err(self):
        print("Testing 4:2:0 error")
        with self.assertRaises(ValueError):
            matchcolors(self.img, self.img, "gamma", fmt=vs.YUV420P8, fname=False, denoise=False)

    def test_yuv_levels_video(self):
        print("Testing YUV levels on video")
        ref = self.vid.std.Levels(min_in=16, max_in=235, min_out=30, max_out=215, planes=0)
        expect = [[30, 215], [16, 240], [16, 240]]
        res = matchcolors(self.vid, ref, fmt=vs.YUV444P8, params="levels", fname=False, denoise=False)
        for i in range(3):
            for j in range(2):
                diff = abs(expect[i][j] - res[i][j])
                self.assertTrue(diff < 10)
        return


if __name__ == "__main__":
    unittest.main()

