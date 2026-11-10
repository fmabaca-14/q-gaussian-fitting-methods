# An Empirical Comparison of q-Gaussian Fitting Methods

Reproduce the **19 figures and three tables** in the manuscript and its two
appendices. This branch contains the paper analysis for solar-wind proton
density, Bitcoin closing prices, and Difficult Run discharge.

## Reproduce

Use Python 3.12. Place the three exports listed in [data/README.md](data/README.md)
in `data/raw/`, then run from this directory:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, you can use a virtual environment from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py
```

The exports are not included. The script verifies the cleaned observations
against the paper snapshots before fitting. Use `--data-dir` to point to
exports stored elsewhere, and `--output-dir` to change the destination.
There is no network access during the analysis.

## Outputs

| Manuscript item | Generated files under `results/` |
| --- | --- |
| Figures 1–3: fluctuations | `figures/figure_01_*` to `figure_03_*` |
| Figures 4–6: fitted PDFs at 100 bins | `figures/figure_04_*` to `figure_06_*` |
| Figure 7: sensitivity of q and b | `figures/figure_07_bin_sensitivity.pdf`, `bin_sensitivity.csv` |
| Figures 8–10: absolute density errors at 100 bins | `figures/figure_08_*` to `figure_10_*` |
| Figure 11: cumulative MAE at 100 bins | `figures/figure_11_cumulative_mae_100bins.pdf` |
| Appendix A (20 bins) and B (50 bins) | `figures/figure_A1_*` to `figure_A4_*`, `figures/figure_B1_*` to `figure_B4_*` |
| Table 1 and Tables A.1 and B.1 | `tables/table_1_100bins`, `tables/table_A1_20bins`, `tables/table_B1_50bins` (CSV and LaTeX) |

The PDF figures can be included with `\includegraphics`. Use the generated
LaTeX tables with `\input` and the `booktabs` package. The CSV tables give
unscaled values. In the LaTeX tables, CvM is displayed in units of 10⁻⁴ and,
in Table 1 only, the MAEs are displayed in units of 10⁻².

`binwise_errors.csv` records both densities and the absolute error for each
bin at 20, 50, and 100 bins. Its cumulative columns use the row's bin number
as the number of bins accumulated from the left or right. `run.json` records
data fingerprints, sample sizes, bin counts, and software versions.

## Numerical protocol

All fits use the normalized q-Gaussian with the location fixed at zero.
The sample mean is not subtracted. The histograms have equal-width bins
covering the complete observed range.

- **Direct PDF:** minimize squared density residuals over occupied bins.
- **q-log:** for q from 1.02 to 2.98 in steps of 0.01, fit a line to
  `ln_q(density)` versus the squared bin centers of occupied bins. The
  squared regression weights are `density**(2*q)`, obtained by propagating a
  constant reference scale in density through the transformation. Recover
  `b = -B / (1 + (1-q)*A)` from intercept A and slope B. Select q by the
  minimum Pearson count discrepancy over **all** bins, with expected counts
  obtained by integrating the normalized candidate density over each bin.
  The model is not renormalized over the observed range.
- **MLE:** minimize the negative log of the product of marginal densities
  using individual observations.
- **CDF:** minimize mean squared differences from the midpoint plotting
  positions of the sorted observations. The analytic CDF uses the exact
  Student-t representation.

Direct PDF and q-log are refitted from 10 to 110 bins in steps of 2.
MLE and CDF are fitted once per dataset. For evaluation, the density
residuals include empty bins; the left and right MAEs average the outermost
10% of bins on each side. The normalized CvM distance uses midpoint
plotting positions and adds `1/(12*N**2)` to the CDF mean squared error.

`reproduce.py` runs the analysis. `src/data.py` handles inputs;
`src/fits.py` contains the estimators and metrics; `src/figures.py` draws the
plots. To check the code without the full datasets:

```sh
python -m unittest discover -s tests -v
```
