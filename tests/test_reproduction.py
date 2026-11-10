import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from src.data import increments
from src.fits import (Histogram, bin_probabilities, cdf, cumulative_mean, density,
                      diagnostics, fit_qlog, fit_unbinned, midpoint_ranks)


class PaperProtocolTests(unittest.TestCase):
    def test_missing_hour_is_not_an_increment(self):
        dates = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 04:00",
                                "2020-01-01 05:00", "2020-01-01 06:00"])
        ts = pd.Series([1., 2., 5., 4., 2.], index=dates)
        time, x, gaps = increments("solar", ts)
        self.assertEqual(gaps, 1)
        self.assertTrue(time.equals(dates[[1, 3, 4]]))
        np.testing.assert_allclose(x, [2/3, -2/9, -2/3])

    def test_pdf_and_integrated_bin_probabilities(self):
        q, b = 1.75, 2.1
        df, scale = (3-q)/(q-1), 1/np.sqrt(b*(3-q))
        edges = np.array([-3., -.7, .4, 2., 5.])
        np.testing.assert_allclose(density(edges, b, q), t.pdf(edges, df=df, scale=scale))
        expected = np.diff(t.cdf(edges, df=df, scale=scale))
        np.testing.assert_allclose(bin_probabilities(edges, b, q), expected, rtol=1e-9)
        self.assertLess(expected.sum(), 1.)

    def test_cdf_recovers_known_centered_quantiles(self):
        q, b = 1.7, 2.3
        x = t.ppf(midpoint_ranks(801), df=(3-q)/(q-1), scale=1/np.sqrt(b*(3-q)))
        fit = fit_unbinned(x, "cdf")
        np.testing.assert_allclose([fit.q, fit.b], [q, b], rtol=5e-4)

    def test_qlog_scores_counts_without_using_the_cdf_fit(self):
        x = np.r_[t.ppf(midpoint_ranks(500), df=5), 8.5, -8.5]
        hist = Histogram.from_sample(x, 50)
        self.assertTrue(np.any(hist.counts == 0))
        with patch("src.fits.cdf", side_effect=AssertionError("q-log must integrate its PDF")):
            fit = fit_qlog(hist)
        expected = len(x)*np.diff(cdf(hist.edges, fit.b, fit.q))
        np.testing.assert_allclose(fit.objective, np.sum((hist.counts-expected)**2 / expected), rtol=1e-8)
        score, _, err = diagnostics(np.sort(x), hist, fit)
        self.assertAlmostEqual(cumulative_mean(err)[-1], score["mae"])
        self.assertAlmostEqual(cumulative_mean(err)[4], score["mae_left"])
        self.assertAlmostEqual(cumulative_mean(err[::-1])[4], score["mae_right"])


if __name__ == "__main__":
    unittest.main()
