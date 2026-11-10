"""Load the three manuscript snapshots and form consecutive increments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path(__file__).resolve().parents[1] / "data" / "snapshots.json"


@dataclass(frozen=True)
class Series:
    name: str
    dates: pd.DatetimeIndex  # Timestamp of the later observation in each pair.
    increments: np.ndarray
    metadata: dict


def read_observations(name, path):
    """Return sorted timestamps and positive source observations."""
    if name == "solar":
        cols = ["year", "day", "hour", "density", "sigma_density"]
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        valid = (frame["density"].notna() & frame["density"].ne(999.9)
                 & frame["sigma_density"].notna() & ~frame["sigma_density"].isin([0, 999.9]))
        frame = frame.loc[valid]
        dates = pd.to_datetime(frame["year"].astype(str) + frame["day"].astype(str).str.zfill(3),
                               format="%Y%j") + pd.to_timedelta(frame["hour"], unit="h")
        values = frame["density"]
    elif name == "bitcoin":
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        dates = pd.to_datetime(frame["timeClose"], utc=True)
        values = frame["close"]
    elif name == "discharge":
        frame = pd.read_csv(path)
        for column, target in (("monitoring_location_id", "USGS-01646000"),
                               ("parameter_code", 60), ("statistic_id", 3)):
            if column in frame and not frame[column].eq(target).all():
                raise ValueError(f"Unexpected {column} in the Difficult Run export: {path}")
        dates = pd.to_datetime(frame["time"])
        values = frame["value"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    ts = pd.Series(pd.to_numeric(values, errors="raise").to_numpy(dtype=float),
                   index=pd.DatetimeIndex(dates)).sort_index()
    if ts.index.has_duplicates or ts.index.hasnans:
        raise ValueError(f"{name}: duplicate or missing timestamps")
    if len(ts) < 4 or not np.all(np.isfinite(ts)) or np.any(ts <= 0):
        raise ValueError(f"{name}: need at least four finite positive observations")
    return ts


def observations_sha256(ts):
    """Fingerprint cleaned timestamps and values independently of CSV formatting."""
    timestamps = ts.index.as_unit("ns").asi8.astype("<i8", copy=False)
    values = ts.to_numpy(dtype="<f8")
    return hashlib.sha256(timestamps.tobytes() + values.tobytes()).hexdigest()


def increments(name, ts):
    """Skip gaps; use symmetric differences for solar/river, log ratios for BTC."""
    cadence = np.timedelta64(1, "h" if name == "solar" else "D")
    valid = np.diff(ts.index.values) == cadence
    previous, current = ts.to_numpy()[:-1][valid], ts.to_numpy()[1:][valid]
    x = (np.log(current / previous) if name == "bitcoin"
         else 2 * (current - previous) / (current + previous))
    if len(x) < 3 or np.ptp(x) == 0:
        raise ValueError(f"{name}: need a nonconstant sample of consecutive increments")
    return ts.index[1:][valid], x, int((~valid).sum())


def load_inputs(directory):
    """Load and verify the archived observations before any optimization."""
    directory = Path(directory)
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [spec["file"] for spec in sources.values() if not (directory / spec["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs in {directory}: {', '.join(missing)}. See data/README.md.")

    data = {}
    for name, spec in sources.items():
        path = directory / spec["file"]
        ts = read_observations(name, path)
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        clean_hash = observations_sha256(ts)
        if clean_hash != spec["observations_sha256"]:
            raise ValueError(f"{name}: observations differ from the paper snapshot. See data/README.md.")
        dates, x, gaps = increments(name, ts)
        if len(x) != spec["n_increments"]:
            raise ValueError(f"{name}: expected {spec['n_increments']} increments, got {len(x)}")
        data[name] = Series(name, dates, x, {
            "file": path.name, "sha256": raw_hash, "observations_sha256": clean_hash,
            "matches_export_bytes": raw_hash == spec["sha256"],
            "n_observations": len(ts), "n_increments": len(x), "skipped_gap_pairs": gaps,
            "first_observation": ts.index[0].isoformat(), "last_observation": ts.index[-1].isoformat(),
            "first_increment": dates[0].isoformat(), "last_increment": dates[-1].isoformat(),
        })
    return data
