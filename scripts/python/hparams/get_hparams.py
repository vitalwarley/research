import pandas as pd
import requests


# TensorBoard URL and Payload
def fetch_session_groups(experiment_name, allowed_statuses, col_params, start_index, slice_size):
    url = "http://localhost:63891/tb/0/data/plugin/hparams/session_groups"
    payload = {
        "experimentName": experiment_name,
        "allowedStatuses": allowed_statuses,
        "colParams": col_params,
        "startIndex": start_index,
        "sliceSize": slice_size,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["sessionGroups"]
    else:
        print(f"Failed to fetch data: {response.status_code}")


# Function to make POST request for detailed metric evaluations
def fetch_metric_evals(session_name, tag):
    metric_evals_url = "http://localhost:63891/tb/0/data/plugin/hparams/metric_evals"
    metric_evals_payload = {"experimentName": "", "sessionName": session_name, "metricName": {"tag": tag, "group": ""}}
    response = requests.post(metric_evals_url, json=metric_evals_payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {tag} in session {session_name}: {response.status_code}")
        return None


# Session groups payload
experiment_name = ""
allowed_statuses = ["STATUS_UNKNOWN", "STATUS_SUCCESS", "STATUS_FAILURE", "STATUS_RUNNING"]
col_params = [
    {"hparam": "batch-size"},
    {"hparam": "clip-gradient"},
    {"hparam": "normalize"},
    {"hparam": "num-epoch"},
    {"metric": {"tag": "acc", "group": ""}},
    {"metric": {"tag": "epoch", "group": ""}},
    {"metric": {"tag": "epoch_acc", "group": ""}},
    {"metric": {"tag": "epoch_auc", "group": ""}},
    {"metric": {"tag": "epoch_loss", "group": ""}},
    {"metric": {"tag": "lr", "group": ""}},
]
start_index = 0
slice_size = 200

# Define the tags to fetch detailed metrics for
tags_to_fetch = ["epoch", "epoch_acc", "epoch_auc", "epoch_loss"]

session_groups = fetch_session_groups(experiment_name, allowed_statuses, col_params, start_index, slice_size)

# Check if the request was successful
if session_groups:
    # Process each session group
    rows = []
    for session in session_groups:
        # Extract session name and hparams
        session_name = session["sessions"][0]["name"]
        hparams = session["hparams"]

        # Fetch and process detailed metrics for the specified tags
        for tag in tags_to_fetch:
            detailed_metrics = fetch_metric_evals(session_name, tag)
            if detailed_metrics:
                print(f"Processing {tag} for session {session_name}")
                for item in detailed_metrics:
                    # Add detailed metric data to rows
                    detailed_row = {"experiment": session_name, "step": item[1], "metric": tag, "value": item[2]}
                    # Add hparams to each detailed row
                    detailed_row.update(hparams)
                    rows.append(detailed_row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv("tensorboard_data.csv", index=False)
    print("Data saved to tensorboard_data.csv")
