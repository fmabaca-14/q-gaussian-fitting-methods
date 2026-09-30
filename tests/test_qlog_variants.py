import unittest

import numpy as np

from src.distributions import q_log
from src.empirical import histogram
from src.qlog_variants import fit_centered_qlog


class CenteredQlogTests(unittest.TestCase):
    def test_selection_uses_unscaled_weighted_sse_and_zero_location(self):
        # An asymmetric sample checks that the estimator does not subtract mu.
        x = np.random.default_rng(14).standard_t(5, size=1500) + .2
        result, scan = fit_centered_qlog(x, bins=50, q_grid=[1.2, 1.5, 1.8, 2.1], return_scan=True)
        self.assertTrue(result.success)
        self.assertEqual(result.mu, 0.)
        best = min((r for r in scan if r['valid']), key=lambda r: r['weighted_sse'])
        self.assertEqual(result.q, best['q'])
        _, _, centers, density, sigma = histogram(x, 50)
        response = q_log(density, result.q)
        line = best['intercept'] + best['slope']*centers**2
        residual = (response-line)/(sigma*density**(-result.q))
        self.assertAlmostEqual(result.objective, float(np.sum(residual**2)), places=8)

    def test_rejects_invalid_grid(self):
        for grid in ([1.], [3.], [], [np.nan]):
            with self.assertRaises(ValueError):
                fit_centered_qlog(np.arange(10.), q_grid=grid)


if __name__ == '__main__':
    unittest.main()
