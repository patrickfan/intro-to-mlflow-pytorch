# MLflow MCP Server (Custom: My Runs)

This document describes `server_custom.py`, a custom MCP server tailored to help you find
runs you worked on and quickly identify the best run by metric (e.g., `val_accuracy`).

Key features:
- Filters runs by your users (default: `8cf`, `patrickfan`).
- Matches common user tag keys (`mlflow.user`, `owner`, `user`).
- Supports best-run queries with metric ordering.
- Includes convenience summary output for best runs.

---

## Setup

### 1) Environment
Ensure you have a Python environment with `mlflow` and `mcp` installed.
Example using your conda env:

```bash
/opt/anaconda3/envs/edm_py311/bin/python -m pip install -r requirements.txt
```

### 2) Required Environment Variables
You must set MLflow connection details and your API key:

```bash
export MLFLOW_TRACKING_URI="https://mlflow.american-science-cloud.org"
export MLFLOW_TRACKING_INSECURE_TLS="true"
export AM_SC_API_KEY="<your-api-key>"
```

Notes:
- `AM_SC_API_KEY` is required for AmSC; requests are sent with `X-Api-Key`.
- If you set `AM_SC_API_KEY` when running `codex mcp add`, it is stored locally in
  `~/.codex/config.toml`.

### 3) Run a Quick Self-Test

```bash
/opt/anaconda3/envs/edm_py311/bin/python mcp_server/server_custom.py --self-test
```

Expected:
- Prints effective config
- Returns a few of your runs

---

## Add to Codex

```bash
/Applications/Codex.app/Contents/Resources/codex mcp add mlflow-amsc-custom       --env MLFLOW_TRACKING_URI=https://mlflow.american-science-cloud.org       --env MLFLOW_TRACKING_INSECURE_TLS=true       --env AM_SC_API_KEY="$AM_SC_API_KEY"       -- /opt/anaconda3/envs/edm_py311/bin/python /Users/8cf/Downloads/intro-to-mlflow-pytorch-main/mcp_server/server_custom.py
```

Verify:

```bash
/Applications/Codex.app/Contents/Resources/codex mcp get mlflow-amsc-custom
```

---

## Tools and Usage

### 1) `list_my_runs`
Lists runs filtered to your users/tags.

```text
list_my_runs(
  experiment_names_csv="",
  usernames_csv="",
  tag_keys_csv="",
  extra_filter="",
  max_results=20,
  order_by="attributes.start_time DESC"
)
```

Example:

```text
Call list_my_runs with {
  "experiment_names_csv": "mnist_cnn_pytorch_exp",
  "max_results": 20
}
```

### 2) `get_best_run`
Returns the single best run by a metric (e.g., `val_accuracy`).

```text
get_best_run(
  experiment_name,
  metric="val_accuracy",
  higher_is_better=true,
  only_mine=true,
  usernames_csv="",
  tag_keys_csv="",
  filter_string="",
  max_results=20
)
```

Example:

```text
Call get_best_run with {
  "experiment_name": "mnist_cnn_pytorch_exp",
  "metric": "val_accuracy"
}
```

### 3) `summarize_best_run`
Same as `get_best_run` but returns a compact summary with params and owner.

```text
Call summarize_best_run with {
  "experiment_name": "mnist_cnn_pytorch_exp",
  "metric": "val_accuracy"
}
```

### 4) `search_runs`
Raw MLflow search for full control.

```text
Call search_runs with {
  "experiment_names_csv": "mnist_cnn_pytorch_exp",
  "filter_string": "attributes.status = 'FINISHED'",
  "order_by": "metrics.val_accuracy DESC",
  "max_results": 50
}
```

### 5) `get_run`
Fetch a single run by run ID.

```text
Call get_run with {"run_id": "<RUN_ID>"}
```

### 6) `describe_server_config`
Shows effective configuration (including whether `AM_SC_API_KEY` is set).

```text
Call describe_server_config from mlflow-amsc-custom
```

---

## How Filters Work

### Default User Filter
The server filters **in code** across usernames and tag keys (to avoid MLflow filter syntax pitfalls):

- Default usernames: `8cf`, `patrickfan`
- Default tag keys: `mlflow.user`, `owner`, `user`

Any run whose tags contain one of these keys with one of these usernames is included.

### Combining Filters
If you pass `extra_filter` (or `filter_string` in `get_best_run`), it is applied **server-side** by MLflow.
The user filter is applied **after** the server returns runs.

Implication: make sure `max_results` (or `MLFLOW_MCP_MAX_RESULTS`) is large enough, or you may
miss older runs that satisfy your user filter.

### Custom Users / Tags
You can override defaults on any call:

```text
Call list_my_runs with {
  "usernames_csv": "8cf",
  "tag_keys_csv": "mlflow.user",
  "max_results": 10
}
```

---

## How to Interpret Results

Every run result includes:
- `run_id`: unique ID
- `run_name`: human readable name (from `mlflow.runName` tag)
- `metrics`: all metrics recorded (e.g., `val_accuracy`, `eval_accuracy`)
- `params`: hyperparameters (batch size, epochs, etc.)
- `tags`: metadata (owner/user, source script, etc.)
- `start_time` / `end_time`: epoch time in **milliseconds**

Tip: convert epoch milliseconds to human time:

```python
from datetime import datetime
datetime.fromtimestamp(start_time / 1000.0)
```

---

## Troubleshooting

**1) `has_am_sc_api_key` is `false`**
- Ensure `AM_SC_API_KEY` is set in your shell before `codex mcp add`.
- Re-run `codex mcp add` and **restart Codex app** to reload config.

**2) No runs returned**
- Check experiment name spelling.
- Your runs may use different tag keys; try `search_runs` with custom filters.
- Set `only_mine=false` in `get_best_run` to inspect all runs.

**3) Best run says “No runs with metric…”**
- Confirm the metric key exists (e.g., `val_accuracy` vs `eval_accuracy`).
- Use `search_runs` to list runs and inspect `metrics` keys.

**4) TLS / network errors**
- Confirm `MLFLOW_TRACKING_URI` is reachable from your machine.
- If needed, keep `MLFLOW_TRACKING_INSECURE_TLS=true`.

---

## Example End-to-End

1) Check config:
```text
Call describe_server_config from mlflow-amsc-custom
```

2) List your runs in one experiment:
```text
Call list_my_runs with {"experiment_names_csv": "mnist_cnn_pytorch_exp", "max_results": 20}
```

3) Find the best run by validation accuracy:
```text
Call summarize_best_run with {"experiment_name": "mnist_cnn_pytorch_exp", "metric": "val_accuracy"}
```
