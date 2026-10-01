"""Fixed location, weighting, gap-safe increments and change-of-unit checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import norm, t
from src.distributions import q_gaussian, q_gaussian_cdf
from src.empirical import fit_histogram, fit_unbinned, histogram, load_increments
from src.gaussian import fit_gaussian_cdf
from src.estimators import fit, histogram as synthetic_histogram

class PaperProtocolTests(unittest.TestCase):
    def test_centered_cdf_known_quantiles_and_units(self):
        q, b = 1.6, 4.0
        nu = (3-q)/(q-1)
        x = t.ppf((np.arange(2000)+.5)/2000, nu)/np.sqrt(b*(3-q))
        a = fit_unbinned(x, "cdf")
        c = fit_unbinned(10*x, "cdf")
        self.assertTrue(a.success and c.success)
        self.assertEqual(a.mu, 0.)
        self.assertAlmostEqual(a.q, q, places=3)
        self.assertAlmostEqual(a.b, b, places=2)
        self.assertAlmostEqual(c.q, a.q, places=3)
        self.assertAlmostEqual(c.b, a.b/100, places=4)
        g = fit_gaussian_cdf(norm.ppf((np.arange(2000)+.5)/2000, scale=2.))
        self.assertTrue(g.success)
        self.assertEqual(g.mu, 0.)
        self.assertAlmostEqual(g.sigma, 2., places=3)

    def test_all_estimators_zero_location_on_shifted_sample(self):
        x = np.random.default_rng(14).standard_t(5, size=2000)+.2
        for method in ("pdf", "qlog", "mle", "cdf"):
            result = fit_histogram(x, 50, method) if method in {"pdf", "qlog"} else fit_unbinned(x, method)
            self.assertTrue(result.success, (method, result.message))
            self.assertEqual(result.mu, 0.)
        self.assertEqual(fit_gaussian_cdf(x).mu, 0.)

    def test_pearson_selection_and_uniform_pdf_objective(self):
        x = np.random.default_rng(20).standard_t(4, size=4000)
        counts, edges, centers, heights, _ = histogram(x, 50)
        direct = fit_histogram(x, 50, "pdf")
        self.assertAlmostEqual(direct.objective,
            np.sum((heights-q_gaussian(centers, direct.b, direct.q))**2), places=12)
        qlog = fit_histogram(x, 50, "qlog")
        expected = len(x)*np.diff(q_gaussian_cdf(edges, qlog.b, qlog.q))
        self.assertAlmostEqual(qlog.objective,
            np.sum((counts-expected)**2/np.maximum(expected, 1e-10)), places=7)
        synthetic = fit(x, "qlog", bins=50, hist=synthetic_histogram(x, 50))
        self.assertEqual(synthetic.q, qlog.q)
        self.assertAlmostEqual(synthetic.b, qlog.b, places=10)

    def test_mean_scaled_river_mean_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            solar, bitcoin, river = (base/name for name in ("s.txt", "b.xls", "r.xls"))
            solar.write_text("2020 1 0 2 1\n2020 1 1 4 1\n2020 1 2 3 1\n2020 1 3 2 1\n")
            bitcoin.write_text("timeClose;close\n2020-01-01T23:59:59.999Z;2\n2020-01-02T23:59:59.999Z;4\n2020-01-03T23:59:59.999Z;3\n2020-01-04T23:59:59.999Z;2\n")
            river.write_text("time,value\n2020-01-01,0\n2020-01-02,4\n2020-01-04,20\n2020-01-05,2\n2020-01-06,0\n")
            data = load_increments(solar, bitcoin, river)["discharge"]
            np.testing.assert_allclose(data["x"], np.array([4., -18., -2.])/5.2)
            self.assertEqual(data["mean_q"], 5.2)
            self.assertEqual(data["n_gaps"], 1)
            self.assertLess(data["x"].min(), -2.)
            self.assertEqual(data["transform"], "mean-scaled")

if __name__ == "__main__":
    unittest.main()
