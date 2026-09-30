import unittest
import numpy as np
from scipy.stats import norm
from src.gaussian import fit_gaussian_cdf


class GaussianTests(unittest.TestCase):
    def test_recovers_gaussian_cdf_location_and_scale(self):
        x=norm.ppf((np.arange(1000)+.5)/1000,loc=.3,scale=2.)
        result=fit_gaussian_cdf(x)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.mu,.3,places=4)
        self.assertAlmostEqual(result.sigma,2.,places=3)

    def test_fit_respects_change_of_units(self):
        x=np.random.default_rng(14).normal(size=1000)
        a=fit_gaussian_cdf(x)
        b=fit_gaussian_cdf(30*x+17)
        self.assertTrue(a.success and b.success)
        self.assertAlmostEqual(b.mu,30*a.mu+17,places=4)
        self.assertAlmostEqual(b.sigma,30*a.sigma,places=4)


if __name__=='__main__':
    unittest.main()
