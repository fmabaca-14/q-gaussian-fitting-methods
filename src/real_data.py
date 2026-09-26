"""Load the original solar wind, Bitcoin, and river discharge time series.

Loading does not calculate increments or assume that observations are iid.
Each function returns a chronologically sorted DataFrame indexed by time.
"""

from pathlib import Path

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_solar_wind(path: str | Path) -> pd.DataFrame:
    """Read headerless OMNI hourly proton density (year, day, hour, np, ErrNp).

    Discard fill values np=999.9 and ErrNp=0 or 999.9, as in the
    exploratory notebook. The error column is retained for later analysis.
    """
    names = ["Año", "Día", "Hs", "np", "ErrNp"]
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=names)
    frame = frame.loc[
        frame["np"].notna()
        & frame["ErrNp"].notna()
        & frame["np"].ne(999.9)
        & ~frame["ErrNp"].isin([0, 999.9])
    ].copy()
    if frame.empty:
        raise ValueError("No valid solar wind observations remain")
    dates = pd.to_datetime(
        frame["Año"].astype(str) + frame["Día"].astype(str).str.zfill(3),
        format="%Y%j",
        errors="raise",
    )
    frame.index = pd.DatetimeIndex(dates + pd.to_timedelta(frame["Hs"], unit="h"), name="Datetime")
    return frame.sort_index()


def load_bitcoin(path: str | Path) -> pd.DataFrame:
    """Read CoinMarketCap semicolon-delimited text, including .xls exports.

    The index uses timeClose and stays timezone aware (UTC). Prices are kept
    in the original units. No log-returns are computed here.
    """
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    _require_columns(frame, {"timeClose", "close"})
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timeClose"], utc=True, errors="raise"), name="Date")
    return frame.sort_index()


def load_discharge(path: str | Path) -> pd.DataFrame:
    """Read a USGS daily discharge CSV (also provided with a .xls suffix).

    Dates are calendar days; value and unit_of_measure retain their source
    meanings. Filter by station upstream if a future export has multiple sites.
    """
    frame = pd.read_csv(path)
    _require_columns(frame, {"time", "value"})
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["time"], errors="raise"), name="Date")
    return frame.sort_index()


def plot_series(solar: pd.DataFrame, bitcoin: pd.DataFrame, discharge: pd.DataFrame):
    """Return a three-panel figure of the untransformed observed series."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, frame, column, title, ylabel, color in zip(
        axes,
        (solar, bitcoin, discharge),
        ("np", "close", "value"),
        ("Viento solar", "Bitcoin", "Caudal"),
        (r"$n_p$", "Precio de cierre", "Caudal (ft³/s)"),
        ("tab:blue", "tab:orange", "tab:green"),
    ):
        ax.plot(frame.index, frame[column], color=color, linewidth=0.7)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()
    return fig, axes
