import unittest

from ProcessProgram.Particle.inspect_co60_active_bins import assign_active_bin


class ActiveBinSamplingTest(unittest.TestCase):
    def test_assigns_co60_active_pixel_bins(self) -> None:
        cases = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5-6",
            6: "5-6",
            7: "7-8",
            8: "7-8",
            9: "9-12",
            12: "9-12",
            13: "13-20",
            20: "13-20",
            21: "21-40",
            40: "21-40",
            41: ">40",
        }
        for active_pixels, expected in cases.items():
            with self.subTest(active_pixels=active_pixels):
                self.assertEqual(assign_active_bin(active_pixels), expected)


if __name__ == "__main__":
    unittest.main()
