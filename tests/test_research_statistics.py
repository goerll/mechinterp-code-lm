import unittest

from scripts.research_statistics import ratio_summary


class StatisticsTests(unittest.TestCase):
    def test_ratio_of_means_differs_from_mean_of_ratios(self):
        rows = [
            {"family": 0, "effect": 1.0, "gap": 1.0},
            {"family": 1, "effect": 1.0, "gap": 9.0},
        ]
        result = ratio_summary(rows, "effect")
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertEqual(result["families"], 2)

    def test_identical_family_totals_have_degenerate_interval(self):
        rows = [{"family": f, "effect": 2.0, "gap": 4.0} for f in range(4)]
        self.assertEqual(ratio_summary(rows, "effect")["ci"], [0.5, 0.5])

    def test_undefined_denominator_is_explicit(self):
        self.assertIsNone(
            ratio_summary([{"family": 0, "effect": 1.0, "gap": 0.0}], "effect")[
                "estimate"
            ]
        )
