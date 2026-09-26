"""Run from repository root: python -m scripts.plot_real_data --solar ... --bitcoin ... --discharge ..."""

import argparse

import matplotlib.pyplot as plt

from src.real_data import load_bitcoin, load_discharge, load_solar_wind, plot_series


def main():
    parser = argparse.ArgumentParser(description="Inspect the three original real-data series")
    parser.add_argument("--solar", required=True, help="Headerless OMNI proton density file")
    parser.add_argument("--bitcoin", required=True, help="CoinMarketCap semicolon-delimited file")
    parser.add_argument("--discharge", required=True, help="USGS daily discharge file")
    parser.add_argument("--output", help="Optional path to save the figure instead of showing it")
    args = parser.parse_args()
    solar = load_solar_wind(args.solar)
    bitcoin = load_bitcoin(args.bitcoin)
    discharge = load_discharge(args.discharge)
    for label, frame in (("Viento solar", solar), ("Bitcoin", bitcoin), ("Caudal", discharge)):
        print(f"{label}: {len(frame)} observaciones, {frame.index.min()} a {frame.index.max()}")
    fig, _ = plot_series(solar, bitcoin, discharge)
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Gráfico guardado en {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
