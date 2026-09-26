"""Parsing checks using only Python's standard test runner."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.real_data import load_bitcoin, load_discharge, load_solar_wind, plot_series


class RealDataTests(unittest.TestCase):
    def test_solar_headerless_and_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "np.txt"
            source.write_text("1980 1 0 4.4 0.2\n1980 1 1 999.9 0.2\n1980 1 2 5.1 0\n1980 2 3 6.0 0.3\n", encoding="utf-8")
            frame = load_solar_wind(source)
            self.assertEqual(frame["np"].tolist(), [4.4, 6.0])
            self.assertEqual(frame.index.tolist(), [pd.Timestamp("1980-01-01"), pd.Timestamp("1980-01-02 03:00")])

    def test_bitcoin_semicolon_bom_and_utc(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bitcoin.xls"
            source.write_text("\ufefftimeClose;close\n2020-01-02T23:59:59.999Z;8\n2020-01-01T23:59:59.999Z;7\n", encoding="utf-8")
            frame = load_bitcoin(source)
            self.assertEqual(frame["close"].tolist(), [7, 8])
            self.assertEqual(str(frame.index.tz), "UTC")

    def test_discharge_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "river.xls"
            source.write_text("time,value,unit_of_measure\n2020-01-02,8,ft^3/s\n2020-01-01,7,ft^3/s\n", encoding="utf-8")
            river = load_discharge(source)
            self.assertEqual(river["value"].tolist(), [7, 8])
            self.assertTrue(river["unit_of_measure"].eq("ft^3/s").all())
            import matplotlib
            matplotlib.use("Agg")
            solar = pd.DataFrame({"np": [4.]}, index=pd.to_datetime(["2020-01-01"]))
            bitcoin = pd.DataFrame({"close": [7.]}, index=pd.to_datetime(["2020-01-01"], utc=True))
            fig, axes = plot_series(solar, bitcoin, river)
            self.assertEqual(len(axes), 3)
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
