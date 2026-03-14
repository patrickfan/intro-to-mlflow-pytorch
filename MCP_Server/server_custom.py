#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mcp.server.fastmcp import FastMCP


AMSC_API_KEY_ENV = "AM_SC_API_KEY"
DEFAULT_TRACKING_URI = "https://mlflow.american-science-cloud.org"
DEFAULT_MAX_RESULTS = 20
DEFAULT_USERNAMES = ["8cf", "patrickfan"]
DEFAULT_TAG_KEYS = ["mlflow.user", "owner", "user"]


def load_examples_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configure_insecure_tls_warnings():
    insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not insecure:
        return

    import urllib3
    from urllib3.exceptions import InsecureRequestWarning

    urllib3.disable_warnings(InsecureRequestWarning)


def enable_amsc_x_api_key():
    if AMSC_API_KEY_ENV not in os.environ:
        return

    import mlflow.utils.rest_utils as rest_utils

    api_key = os.environ[AMSC_API_KEY_ENV]
    original_http_request = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        headers = dict(kwargs.get("extra_headers") or {})
        if kwargs.get("headers") is not None:
            headers.update(dict(kwargs["headers"]))
        headers["X-Api-Key"] = api_key
        kwargs["extra_headers"] = headers
        kwargs.pop("headers", None)
        return original_http_request(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched


def parse_experiment_ids(experiment_names_csv: str) -> list[str]:
    client = MlflowClient()
    names = [n.strip() for n in experiment_names_csv.split(",") if n.strip()]
    if not names:
        return []

    ids = []
    for name in names:
        exp = client.get_experiment_by_name(name)
        if exp:
            ids.append(exp.experiment_id)
    return ids


def run_to_dict(run):
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "experiment_id": run.info.experiment_id,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def experiment_to_dict(exp):
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "lifecycle_stage": exp.lifecycle_stage,
        "artifact_location": exp.artifact_location,
        "creation_time": getattr(exp, "creation_time", None),
        "last_update_time": getattr(exp, "last_update_time", None),
    }


def parse_view_type(view_type: str) -> ViewType:
    value = (view_type or "").strip().lower()
    if value == "all":
        return ViewType.ALL
    if value == "deleted_only":
        return ViewType.DELETED_ONLY
    return ViewType.ACTIVE_ONLY


def effective_max_results(max_results: int) -> int:
    if max_results and max_results > 0:
        return max_results
    return int(os.environ.get("MLFLOW_MCP_MAX_RESULTS", str(DEFAULT_MAX_RESULTS)))


def parse_csv(value: str, fallback: list[str]) -> list[str]:
    if value:
        items = [v.strip() for v in value.split(",") if v.strip()]
        if items:
            return items
    return list(fallback)


def format_tag_key(tag: str) -> str:
    # MLflow requires backticks for tag keys with dots or other special chars.
    safe = tag.replace("_", "")
    if safe.isalnum():
        return tag
    return f"`{tag}`"


def build_user_filter(usernames_csv: str = "", tag_keys_csv: str = "") -> str:
    usernames = parse_csv(usernames_csv, DEFAULT_USERNAMES)
    tag_keys = parse_csv(tag_keys_csv, DEFAULT_TAG_KEYS)
    clauses = []
    for username in usernames:
        for tag in tag_keys:
            key = format_tag_key(tag)
            clauses.append(f"tags.{key} = '{username}'")
    return "(" + " OR ".join(clauses) + ")" if clauses else ""


def combine_filters(base: str, extra: str) -> str:
    base = (base or "").strip()
    extra = (extra or "").strip()
    if base and extra:
        return f"({base}) AND ({extra})"
    return base or extra


def run_matches_user(run, usernames: list[str], tag_keys: list[str]) -> bool:
    tags = run.data.tags or {}
    for key in tag_keys:
        if tags.get(key) in usernames:
            return True
    return False


load_examples_env()
os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
configure_insecure_tls_warnings()
enable_amsc_x_api_key()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

mcp = FastMCP("mlflow-experiments-mcp-custom")


@mcp.tool()
def list_experiments(max_results: int = DEFAULT_MAX_RESULTS, view_type: str = "active_only") -> list[dict]:
    """List MLflow experiments.

    Args:
        max_results: Max experiments to return.
        view_type: active_only | deleted_only | all
    """
    client = MlflowClient()
    exps = client.search_experiments(
        view_type=parse_view_type(view_type),
        max_results=effective_max_results(max_results),
    )
    return [experiment_to_dict(exp) for exp in exps]


@mcp.tool()
def list_my_runs(
    experiment_names_csv: str = "",
    usernames_csv: str = "",
    tag_keys_csv: str = "",
    extra_filter: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    order_by: str = "attributes.start_time DESC",
) -> list[dict]:
    """List runs filtered to your users/tags.

    Args:
        experiment_names_csv: Comma-separated experiment names. Empty uses all active experiments.
        usernames_csv: Comma-separated usernames. Empty uses defaults in server.
        tag_keys_csv: Comma-separated tag keys to match (e.g. mlflow.user, owner, user).
        extra_filter: Additional MLflow filter expression to AND with the user filter.
        max_results: Max runs to return.
        order_by: Single MLflow order expression.
    """
    client = MlflowClient()
    experiment_ids = parse_experiment_ids(experiment_names_csv)
    if not experiment_ids:
        experiment_ids = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=1000)]

    usernames = parse_csv(usernames_csv, DEFAULT_USERNAMES)
    tag_keys = parse_csv(tag_keys_csv, DEFAULT_TAG_KEYS)

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=extra_filter,
        max_results=effective_max_results(max_results),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[order_by] if order_by else None,
    )
    return [run_to_dict(r) for r in runs if run_matches_user(r, usernames, tag_keys)]


@mcp.tool()
def search_runs(
    experiment_names_csv: str = "",
    filter_string: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    order_by: str = "attributes.start_time DESC",
) -> list[dict]:
    """Search runs with a raw MLflow filter string.

    Args:
        experiment_names_csv: Comma-separated experiment names. Empty uses all active experiments.
        filter_string: MLflow run filter expression.
        max_results: Max runs to return.
        order_by: Single MLflow order expression.
    """
    client = MlflowClient()
    experiment_ids = parse_experiment_ids(experiment_names_csv)
    if not experiment_ids:
        experiment_ids = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=1000)]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=effective_max_results(max_results),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[order_by] if order_by else None,
    )
    return [run_to_dict(r) for r in runs]


@mcp.tool()
def get_best_run(
    experiment_name: str,
    metric: str = "val_accuracy",
    higher_is_better: bool = True,
    only_mine: bool = True,
    usernames_csv: str = "",
    tag_keys_csv: str = "",
    filter_string: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict:
    """Find the best run in an experiment by a metric.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric key to sort by (e.g. val_accuracy, eval_accuracy).
        higher_is_better: If True, sort descending; if False, ascending.
        only_mine: If True, restrict to runs matching usernames/tag keys.
        usernames_csv: Comma-separated usernames. Empty uses defaults in server.
        tag_keys_csv: Comma-separated tag keys to match (e.g. mlflow.user, owner, user).
        filter_string: Optional MLflow run filter expression.
        max_results: Max runs to consider (defaults to MLFLOW_MCP_MAX_RESULTS).
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return {"error": f"Experiment not found: {experiment_name}"}

    if not metric or not metric.strip():
        return {"error": "Metric name must be non-empty."}
    metric = metric.strip()
    order = "DESC" if higher_is_better else "ASC"
    order_by = f"metrics.{metric} {order}"

    usernames = parse_csv(usernames_csv, DEFAULT_USERNAMES)
    tag_keys = parse_csv(tag_keys_csv, DEFAULT_TAG_KEYS)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=effective_max_results(max_results),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[order_by],
    )

    best_run = None
    best_value = None
    for run in runs:
        if only_mine and not run_matches_user(run, usernames, tag_keys):
            continue
        if metric in run.data.metrics:
            value = run.data.metrics[metric]
            if best_value is None:
                best_value = value
                best_run = run
            elif higher_is_better and value > best_value:
                best_value = value
                best_run = run
            elif not higher_is_better and value < best_value:
                best_value = value
                best_run = run

    if best_run is not None:
        payload = run_to_dict(best_run)
        payload["best_metric"] = {
            "name": metric,
            "value": best_value,
            "higher_is_better": higher_is_better,
        }
        return payload

    return {"error": f"No runs with metric '{metric}' found in experiment '{experiment_name}'."}


@mcp.tool()
def summarize_best_run(
    experiment_name: str,
    metric: str = "val_accuracy",
    higher_is_better: bool = True,
    only_mine: bool = True,
    usernames_csv: str = "",
    tag_keys_csv: str = "",
    filter_string: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict:
    """Return a compact summary of the best run.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric key to sort by.
        higher_is_better: If True, sort descending; if False, ascending.
        only_mine: If True, restrict to runs matching usernames/tag keys.
        usernames_csv: Comma-separated usernames. Empty uses defaults in server.
        tag_keys_csv: Comma-separated tag keys to match (e.g. mlflow.user, owner, user).
        filter_string: Optional MLflow run filter expression.
        max_results: Max runs to consider (defaults to MLFLOW_MCP_MAX_RESULTS).
    """
    result = get_best_run(
        experiment_name=experiment_name,
        metric=metric,
        higher_is_better=higher_is_better,
        only_mine=only_mine,
        usernames_csv=usernames_csv,
        tag_keys_csv=tag_keys_csv,
        filter_string=filter_string,
        max_results=max_results,
    )
    if "error" in result:
        return result

    metric_info = result.get("best_metric", {})
    tags = result.get("tags", {})
    owner = tags.get("owner") or tags.get("user") or tags.get("mlflow.user")
    return {
        "run_id": result.get("run_id"),
        "run_name": result.get("run_name"),
        "experiment_id": result.get("experiment_id"),
        "status": result.get("status"),
        "best_metric": metric_info,
        "params": result.get("params", {}),
        "owner": owner,
        "start_time": result.get("start_time"),
        "end_time": result.get("end_time"),
    }


@mcp.tool()
def get_run(run_id: str) -> dict:
    """Get details for a single MLflow run by run_id."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run_to_dict(run)


@mcp.tool()
def get_experiment(experiment_name: str) -> dict:
    """Get a single experiment by name."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return {"error": f"Experiment not found: {experiment_name}"}
    return experiment_to_dict(exp)


@mcp.tool()
def describe_server_config() -> dict:
    """Show effective MLflow connection config used by this MCP server."""
    return {
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        "mlflow_tracking_insecure_tls": os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "true"),
        "has_am_sc_api_key": AMSC_API_KEY_ENV in os.environ and bool(os.environ.get(AMSC_API_KEY_ENV)),
        "default_max_results": int(os.environ.get("MLFLOW_MCP_MAX_RESULTS", str(DEFAULT_MAX_RESULTS))),
        "default_usernames": list(DEFAULT_USERNAMES),
        "default_tag_keys": list(DEFAULT_TAG_KEYS),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow MCP server (custom)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick MLflow query and exit (no MCP server start).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max results for --self-test (default: 10).",
    )
    args = parser.parse_args()

    if args.self_test:
        cfg = describe_server_config()
        print("MCP config:", cfg)
        rows = list_my_runs(max_results=args.max_results)
        print(f"Self-test OK. Fetched {len(rows)} run(s).")
        if rows:
            print("Sample run:", rows[0].get("run_name"))
        sys.exit(0)

    print(
        "Starting MLflow MCP server (custom) on stdio. This process waits for an MCP client and may appear idle.",
        file=sys.stderr,
        flush=True,
    )
    mcp.run()
