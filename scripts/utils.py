import io
import urllib.parse

import pandas as pd
import requests


def fetch_scalar_data(url, run, tag):
    """Fetch scalar data for a specific run and tag from TensorBoard."""
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
    """Fetch all runs from TensorBoard."""
    try:
        response = requests.get(f"{url}/data/runs")
        response.raise_for_status()
        runs = response.json()
        filtered_runs = [run for run in runs if ".guild" not in run]
        print(f"Found {len(filtered_runs)} runs after filtering")
        return filtered_runs
    except requests.RequestException as e:
        print(f"Failed to fetch runs: {e}")
        return []


def fetch_all_tags(url, runs):
    """Fetch all available metric tags by checking the first data point of each run."""
    all_tags = {}
    for run in runs:
        try:
            response = requests.get(f"{url}/data/plugin/scalars/tags/{run}")
            response.raise_for_status()
            all_tags[run] = response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch tags for run {run}: {e}")
    return all_tags


def rename_runs(runs):
    """Prompt user to rename runs."""
    renamed_runs = {}
    for run in runs:
        new_name = input(f"Enter new name for run '{run}' (press Enter to keep current name): ").strip()
        renamed_runs[run] = new_name if new_name else run
    return renamed_runs


def load_data(url, metrics, renamed_runs):
    """Load data for given metrics and runs."""
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
