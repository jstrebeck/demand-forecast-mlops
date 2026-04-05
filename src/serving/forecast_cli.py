"""
CLI tool to hit the forecast endpoint and plot results in the terminal.

Usage:
    python -m src.serving.forecast_cli HVP-24242XM
    python -m src.serving.forecast_cli HVP-24242XM --weeks 24
    python -m src.serving.forecast_cli HVP-24242XM MP13-16252 --weeks 12
"""

import argparse
import sys

import plotext as plt
import requests

DEFAULT_URL = "http://localhost:8000"


def fetch_forecast(base_url: str, sku: str, weeks: int) -> dict:
    resp = requests.post(
        f"{base_url}/predict/forecast",
        json={"sku": sku, "weeks": weeks},
    )
    if resp.status_code == 404:
        print(f"SKU '{sku}' not found", file=sys.stderr)
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def plot_single(data: dict):
    dates = [w["date"] for w in data["forecast"]]
    qtys = [w["predicted_qty"] for w in data["forecast"]]

    plt.clear_figure()
    plt.date_form("Y-m-d")
    plt.plot(dates, qtys, marker="braille")
    plt.title(f"Demand Forecast: {data['sku']}")
    plt.xlabel("Week")
    plt.ylabel("Predicted Qty")
    plt.show()


def plot_multi(results: list[dict]):
    plt.clear_figure()
    plt.date_form("Y-m-d")
    for data in results:
        dates = [w["date"] for w in data["forecast"]]
        qtys = [w["predicted_qty"] for w in data["forecast"]]
        plt.plot(dates, qtys, label=data["sku"], marker="braille")
    plt.title("Demand Forecast Comparison")
    plt.xlabel("Week")
    plt.ylabel("Predicted Qty")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Forecast demand and plot in terminal")
    parser.add_argument("skus", nargs="+", help="One or more SKU codes")
    parser.add_argument("--weeks", type=int, default=12, help="Weeks to forecast (default: 12)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Serving API base URL")
    args = parser.parse_args()

    results = []
    for sku in args.skus:
        data = fetch_forecast(args.url, sku, args.weeks)
        results.append(data)

    if len(results) == 1:
        plot_single(results[0])
    else:
        plot_multi(results)


if __name__ == "__main__":
    main()
