"""Check ECDF jump handling and probability conventions for both tails."""

import unittest

import numpy as np
from scipy.stats import t

from src.empirical import EmpiricalFit
from src.empirical_scores import body_scores, exceedance_curve, tail_scores, tail_thresholds


class EmpiricalScoreTests(unittest.TestCase):
    def setUp(self):
        self.fit = EmpiricalFit("mle", b=2.0, q=1.5, mu=0.0, success=True)

    def test_cdf_scores_use_right_ecdf_and_both_jump_sides(self):
        x = np.array([-1., 0., 0., 1.])
        scores = body_scores(x, self.fit, bins=4)
        nu, scale = 3.0, 1 / np.sqrt(3.0)
        f = t.cdf(x / scale, nu)
        right = np.array([.25, .75, .75, 1.])
        left = np.array([0., .25, .25, .75])
        self.assertAlmostEqual(scores["dic_dfn"], np.mean((f - right)**2))
        self.assertAlmostEqual(scores["ks"], max(np.max(abs(f-left)), np.max(abs(f-right))))

    def test_threshold_errors_and_exceedance_use_strict_right_tail(self):
        x = np.array([-2., -1., 0., 1., 2.])
        thresholds = ((10., -1.), (90., 1.))
        rows, mae = tail_scores(x, self.fit, thresholds)
        self.assertEqual(rows[0]["observed_probability"], .4)
        self.assertEqual(rows[1]["observed_probability"], .2)
        self.assertAlmostEqual(mae, np.mean([r["abs_error_pp"] for r in rows]))
        grid, observed, predicted = exceedance_curve(x, self.fit, percentiles=[75])
        self.assertAlmostEqual(observed[0], .2)
        self.assertAlmostEqual(predicted[0], t.sf(grid[0] * np.sqrt(3), 3))

    def test_thresholds_are_shared_empirical_quantiles(self):
        values = tail_thresholds(np.arange(100.))
        self.assertEqual([p for p, _ in values], [1, 5, 10, 90, 95, 99])


if __name__ == "__main__":
    unittest.main()
