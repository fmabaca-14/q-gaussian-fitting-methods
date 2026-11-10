# Input data

Place the three original exports below in `data/raw/`, or pass their directory
with `python reproduce.py --data-dir /path/to/exports`. The files are sourced
from the providers below and are not bundled with the code. The `.xls` filenames
are historical names for delimited **text**, so do not resave them in Excel.

| Export filename | Source and fields | First and last observations used | Increments |
|---|---|---|---:|
| `1980-2024 np.txt` | [NASA OMNIWeb hourly OMNI 2](https://omniweb.gsfc.nasa.gov/form/dx1.html): select **Proton Density, n/cc** and **Sigma-Np** for 1980-01-01 to 2024-12-31. Headerless space-delimited columns: year, day of year, hour, density, sigma. | 1980-01-01 to 2024-12-31 | 296,197 |
| `Bitcoin_15_10_2013-14_12_2013_historical_data_coinmarketcap.xls` | [CoinMarketCap Bitcoin historical data](https://coinmarketcap.com/currencies/bitcoin/historical-data/): daily closing prices in the original semicolon-delimited export, columns `timeClose` (UTC) and `close`. The date in the filename is not the date range of the file. | 2012-11-11 to 2026-01-01 | 4,799 |
| `daily-data-mean.xls` | [USGS Difficult Run near Great Falls, VA, 01646000](https://waterdata.usgs.gov/monitoring-location/USGS-01646000/): daily mean discharge, parameter `00060`, statistic `00003`, CSV columns `time` and `value`. | 2016-05-04 to 2026-05-03 | 3,651 |

The pipeline sorts timestamps, removes the OMNI fill values (density or sigma
`999.9`, and sigma `0`), and keeps only pairs separated by exactly one hour
(OMNI) or one day (Bitcoin and discharge). The increments are
`2*(next-current)/(next+current)` for density and discharge, and
`log(next/current)` for Bitcoin. Their timestamps are the later observations
in each pair. The observations and increments are used without subtracting
their sample means.

`snapshots.json` identifies the exact exports (SHA-256) and the cleaned
observations (SHA-256 of the sorted timestamps and values). Equivalent exports
with a different line ending or column order are accepted when the parsed
observations match. Providers can revise historical values and file formats;
if a new export does not match, request the exact archived inputs from the
corresponding author. The run stops on a mismatch instead of silently using a
different sample. `results/run.json` records the checksums used in each run.
