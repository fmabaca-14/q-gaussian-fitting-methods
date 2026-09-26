"""Checks for chronology, gaps and free-location empirical fits."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import t

from src.empirical import fit_histogram, load_increments


class EmpiricalTests(unittest.TestCase):
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
        qlog = fit_histogram(sample, 80, "qlog")
        self.assertTrue(direct.success and qlog.success)
        self.assertLess(abs(direct.mu-mu), 0.08)
        self.assertLess(abs(qlog.mu-mu), 0.12)
        self.assertLess(abs(direct.q-q), 0.25)
        self.assertLess(abs(qlog.q-q), 0.35)


if __name__ == "__main__":
    unittest.main()
