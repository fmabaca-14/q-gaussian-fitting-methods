"""Checks for chronology, gaps and free-location empirical fits."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.stats import t

from src.empirical import fit_histogram, load_increments, histogram


class EmpiricalTests(unittest.TestCase):
    def test_qlog_selects_regression_minimum_without_count_space_evaluation(self):
        x = np.random.default_rng(14).standard_t(5, size=3000) + .1
        grid = [1.2, 1.5, 1.8, 2.1]
        _, edges, centers, y, sigma = histogram(x, 50)
        scale = np.std(centers)
        design = np.column_stack((np.ones(len(centers)), centers/scale, (centers/scale)**2))
        scores = []
        for q in grid:
            response = np.expm1((1-q)*np.log(y))/(1-q)
            weights = y**q/sigma
            coef, _, _, _ = np.linalg.lstsq(design*weights[:,None], response*weights, rcond=None)
            mu = -coef[1]*scale/(2*coef[2])
            if coef[2] < 0 and edges[0] < mu < edges[-1]:
                scores.append((float(np.sum(((response-design@coef)*weights)**2)), q))
        score, q = min(scores)
        with patch('src.empirical.q_gaussian_cdf', side_effect=AssertionError('CDF must not select q')):
            result = fit_histogram(x, 50, 'qlog', q_grid=grid)
        self.assertTrue(result.success)
        self.assertEqual(result.q, q)
        self.assertAlmostEqual(result.objective, score, places=7)

    def test_legacy_qlog_selection_is_explicit(self):
        x = np.random.default_rng(14).standard_t(5, size=1500)
        result = fit_histogram(x, 50, 'qlog', q_grid=[1.2,1.5,1.8], qlog_selection='pearson_counts')
        self.assertTrue(result.success)
        self.assertIn('Pearson',result.message)

    def test_original_increment_definitions_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            solar, bitcoin, river = (base / name for name in ("solar.txt", "bitcoin.xls", "river.xls"))
            solar.write_text("2020 1 0 2 1\n2020 1 1 4 1\n2020 1 3 8 1\n"
                             "2020 1 4 4 1\n2020 1 5 2 1\n", encoding="utf-8")
            bitcoin.write_text("timeClose;close\n2020-01-01T23:59:59.999Z;2\n"
                               "2020-01-02T23:59:59.999Z;4\n"
                               "2020-01-04T23:59:59.999Z;8\n"
                               "2020-01-05T23:59:59.999Z;4\n"
                               "2020-01-06T23:59:59.999Z;2\n", encoding="utf-8")
            river.write_text("time,value\n2020-01-01,2\n2020-01-02,4\n"
                             "2020-01-04,8\n2020-01-05,4\n2020-01-06,2\n", encoding="utf-8")
            data = load_increments(solar, bitcoin, river)
            np.testing.assert_allclose(data["solar"]["x"], [2/3, -2/3, -2/3])
            np.testing.assert_allclose(data["bitcoin"]["x"],
                                       [np.log(2), -np.log(2), -np.log(2)])
            np.testing.assert_allclose(data["discharge"]["x"], [2/3, -2/3, -2/3])
            self.assertTrue(all(data[k]["n_gaps"] == 1 for k in data))

    def test_qlog_can_recover_nonzero_location_on_synthetic_data(self):
        rng = np.random.default_rng(14)
        q, b, mu = 1.6, 4.0, 0.3
        nu = (3-q)/(q-1)
        sample = mu + t.rvs(df=nu, scale=1/np.sqrt(b*(3-q)),
                            size=12000, random_state=rng)
        direct = fit_histogram(sample, 80, "pdf")
        qlog = fit_histogram(sample, 80, "qlog", qlog_selection="pearson_counts")
        self.assertTrue(direct.success and qlog.success)
        self.assertLess(abs(direct.mu-mu), 0.08)
        self.assertLess(abs(qlog.mu-mu), 0.12)
        self.assertLess(abs(direct.q-q), 0.25)
        self.assertLess(abs(qlog.q-q), 0.35)


if __name__ == "__main__":
    unittest.main()
