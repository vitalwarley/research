import argparse
import io
import urllib.parse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

sns.set_style("darkgrid")


def fetch_scalar_data(url, run, tag):
    encoded_run = urllib.parse.quote(run, safe="")
    encoded_tag = urllib.parse.quote(tag, safe="")
    full_url = f"{url}/data/plugin/scalars/scalars?run={encoded_run}&tag={encoded_tag}&format=csv"
    try:
        response = requests.get(full_url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except requests.RequestException as e:
        print(f"Failed to fetch data for run '{run}', tag '{tag}': {e}")
        print(f"URL attempted: {full_url}")
        return None


def fetch_runs(url):
    try:
        response = requests.get(f"{url}/data/runs")
        response.raise_for_status()
        runs = response.json()
        # Filter out runs with ".guild" in the name
        filtered_runs = [run for run in runs if ".guild" not in run]
        print(f"Found {len(filtered_runs)} runs after filtering")
        return filtered_runs
    except requests.RequestException as e:
        print(f"Failed to fetch runs: {e}")
        return []


def rename_runs(runs):
    renamed_runs = {}
    for run in runs:
        new_name = input(f"Enter new name for run '{run}' (press Enter to keep current name): ").strip()
        renamed_runs[run] = new_name if new_name else run
    return renamed_runs


def load_data(url, metrics, renamed_runs):
    data = []
    for original_run, new_run_name in renamed_runs.items():
        for metric in metrics:
            metric_data = fetch_scalar_data(url, original_run, metric)
            if metric_data is not None:
                print(f"{metric} fetched for run {new_run_name}")
                metric_data["run"] = new_run_name
                metric_data["metric"] = metric
                data.append(metric_data)
            else:
                print(f"No data for {metric} in run {new_run_name}")

    if not data:
        print("No data was successfully fetched.")
        return None

    return pd.concat(data, ignore_index=True)


def plot_metrics(df, metrics, output_dir):
    if df is None or df.empty:
        print("No data to plot.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_data = df[df["metric"] == metric]

        if metric_data.empty:
            print(f"No data available for metric: {metric}")
            continue

        for run in metric_data["run"].unique():
            run_data = metric_data[metric_data["run"] == run]
            plt.plot(run_data["Step"], run_data["Value"], label=run)

        plt.title(f"{metric} over time")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metric = metric.replace("/", "_")
        plt.savefig(f"{output_dir}/{metric}_over_time_{now}.png")
        plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze TensorBoard metrics")
    parser.add_argument("--url", type=str, required=True, help="TensorBoard URL")
    parser.add_argument("--metrics", nargs="+", required=True, help="Metrics to analyze")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_arguments()

    runs = fetch_runs(args.url)
    renamed_runs = rename_runs(runs)

    df = load_data(args.url, args.metrics, renamed_runs)

    plot_metrics(df, args.metrics, args.output_dir)

    if df is not None and not df.empty:
        print(f"Plots saved in {args.output_dir}")
    else:
        print("No plots were generated due to lack of data.")


if __name__ == "__main__":
    main()
