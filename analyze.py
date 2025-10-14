#!/usr/bin/env python3
"""
Load a code-quality JSON report into a pandas DataFrame, create aggregate metrics,
and append them to a CSV stored in a Hugging Face Space.

Usage:
    # 1) create the report (example)
    python main.py <path-to-src> -o report.json

    # 2) analyze the report and push row to Space CSV
    python analyze.py result.json \
        --src-path /path/to/src \
        --hub-repo your-username/your-space-name \
        --hub-file metrics.csv \
        --hub-branch main
"""

import argparse
import datetime as dt
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub import CommitOperationAdd


BASE_FIELDS = [
    "name",
    "nodetype",
    "path",
    "qualname",
    "lineno",
    "end_lineno",
    "docstring",
]

METRIC_FIELDS = [
    "metrics.lines",
    "metrics.statements",
    "metrics.expressions",
    "metrics.expression_statements",
    "metrics.cyclomatic_complexity",
    "metrics.parameters",
    "metrics.type_coverage",
    "metrics.todo_comments",
    "metrics.duplication.score",
    "metrics.duplication.other",
    "metrics.duplication.lines_other",
]


def _iter_nodes(node: dict[str, Any], parent: Optional[tuple[str, ...]] = None) -> Iterator[dict[str, Any]]:
    parent = parent or tuple()

    row: dict[str, Any] = {
        k: node.get(k)
        for k in [
            "name",
            "nodetype",
            "path",
            "qualname",
            "lineno",
            "end_lineno",
            "docstring",
        ]
    }

    if not row.get("qualname"):
        row["qual_or_name"] = row.get("qualname") or row.get("name")
    else:
        row["qual_or_name"] = row["qualname"]

    m = node.get("metrics") or {}
    row["metrics.lines"] = m.get("lines")
    row["metrics.statements"] = m.get("statements")
    row["metrics.expressions"] = m.get("expressions")
    row["metrics.expression_statements"] = m.get("expression_statements")
    row["metrics.cyclomatic_complexity"] = m.get("cyclomatic_complexity")
    row["metrics.parameters"] = m.get("parameters")
    row["metrics.type_coverage"] = m.get("type_coverage")
    row["metrics.todo_comments"] = m.get("todo_comments")

    dup = (m or {}).get("duplication") or {}
    row["metrics.duplication.score"] = dup.get("score")
    row["metrics.duplication.other"] = dup.get("other")
    row["metrics.duplication.lines_other"] = dup.get("lines_other")

    row["has_metrics"] = bool(m)
    row["is_directory"] = row.get("nodetype") == "directory"
    row["is_file"] = row.get("nodetype") == "file"

    yield row

    for child in node.get("children", []) or []:
        yield from _iter_nodes(child, parent)


def load_report_to_df(path: str | Path, *, only_leaves: bool = False) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(_iter_nodes(data))
    df = pd.DataFrame.from_records(
        rows,
        columns=BASE_FIELDS + ["qual_or_name", "has_metrics", "is_directory", "is_file"] + METRIC_FIELDS,
    )

    if only_leaves:
        df = df[df["has_metrics"]].reset_index(drop=True)

    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df["metrics.lines"] = df["metrics.lines"].fillna(0.0).astype(int)
    df["metrics.statements"] = df["metrics.statements"].fillna(0.0).astype(int)
    df["metrics.expressions"] = df["metrics.expressions"].fillna(0.0).astype(int)
    df["metrics.expression_statements"] = df["metrics.expression_statements"].fillna(0.0).astype(int)
    df["metrics.cyclomatic_complexity"] = df["metrics.cyclomatic_complexity"].fillna(0.0).astype(int)
    df["metrics.parameters"] = df["metrics.parameters"].fillna(0.0).astype(int)
    df["metrics.todo_comments"] = df["metrics.todo_comments"].fillna(0.0).astype(int)
    df["metrics.duplication.lines_other"] = df["metrics.duplication.lines_other"].fillna(0.0).astype(int)
    df["metrics.duplication.score"] = df["metrics.duplication.score"].fillna(0.0).astype(float)
    return df


def cloc_metrics(path: Path) -> dict[str, int]:
    """Collect general metrics by calling cloc"""
    result: dict[str, int] = {}
    try:
        proc = subprocess.run(
            ["cloc", "--json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        cloc_data = json.loads(proc.stdout)
        if "Python" in cloc_data:
            py_data = cloc_data["Python"]
            result["files"] = py_data.get("nFiles", 0)
            result["lines blank"] = py_data.get("blank", 0)
            result["lines comment"] = py_data.get("comment", 0)
            result["lines code"] = py_data.get("code", 0)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    return result


def aggregate_metrics(df: pd.DataFrame, date: str) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {}
    result["date"] = date
    result["docstring coverage"] = float((df["docstring"].str.len() > 0).mean().round(4))
    result["docstring missing"] = int((df["docstring"].str.len() == 0).sum())

    result["lines mean"] = float(df["metrics.lines"].mean().round(4))
    result["lines max"] = int(df["metrics.lines"].max())
    result["lines 90th-percentile"] = int(df["metrics.lines"].quantile(0.9))

    result["statements mean"] = float(df["metrics.statements"].mean().round(4))
    result["statements max"] = int(df["metrics.statements"].max())
    result["statements 90th-percentile"] = int(df["metrics.statements"].quantile(0.9))

    result["expressions mean"] = float(df["metrics.expressions"].mean().round(4))
    result["expressions max"] = int(df["metrics.expressions"].max())
    result["expressions 90th-percentile"] = int(df["metrics.expressions"].quantile(0.9))

    result["cyclomatic_complexity mean"] = float(df["metrics.cyclomatic_complexity"].mean().round(4))
    result["cyclomatic_complexity max"] = int(df["metrics.cyclomatic_complexity"].max())
    result["cyclomatic_complexity 90th-percentile"] = int(df["metrics.cyclomatic_complexity"].quantile(0.9))

    result["parameters mean"] = float(df["metrics.parameters"].mean().round(4))
    result["parameters max"] = int(df["metrics.parameters"].max())
    result["parameters 90th-percentile"] = int(df["metrics.parameters"].quantile(0.9))

    result["type_coverage mean"] = float(df["metrics.type_coverage"].mean().round(4))
    result["type_coverage min"] = int(df["metrics.type_coverage"].min())
    result["type_coverage 50th-percentile"] = int(df["metrics.type_coverage"].quantile(0.5))

    result["todo_comments total"] = int(df["metrics.todo_comments"].sum())

    result["duplication.score mean"] = float(df["metrics.duplication.score"].mean().round(4))
    result["duplication.score max"] = float(df["metrics.duplication.score"].max())
    result["duplication.score 90th-percentile"] = float(df["metrics.duplication.score"].quantile(0.9))
    result["duplication.score 50th-percentile"] = float(df["metrics.duplication.score"].quantile(0.5))
    result["duplication.duplicated-lines total"] = int(
        (df["metrics.lines"] * df["metrics.duplication.score"]).sum().round(0).astype(int)
    )

    return result


def _download_space_csv_or_empty(
    repo_id: str,
    path_in_repo: str,
    *,
    repo_type: str = "space",
    revision: str = "main",
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Try to download a CSV from a Space. If missing or empty, return an empty DataFrame."""
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=None,
        )
    except EntryNotFoundError:
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_metrics_to_hub_csv(
    repo_id: str,
    path_in_repo: str,
    row: dict[str, Any],
    *,
    repo_type: str = "space",
    branch: str = "main",
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """
    Append a row of metrics to a CSV in a Hugging Face repo (Space by default).
    Returns the commit OID (sha) for reference.
    """
    api = HfApi(token=token)

    # 1) Load existing CSV (or get empty df)
    df_existing = _download_space_csv_or_empty(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        revision=branch,
        token=token,
    )

    # 2) Append the new row; align columns (union)
    df_new = pd.DataFrame([row])
    all_cols = list(dict.fromkeys([*df_existing.columns.tolist(), *df_new.columns.tolist()]))
    if df_existing.empty:
        df_updated = df_new.reindex(columns=all_cols)
    else:
        df_updated = pd.concat(
            [df_existing.reindex(columns=all_cols), df_new.reindex(columns=all_cols)],
            ignore_index=True,
        )

    # 3) Serialize to CSV bytes (UTF-8)
    csv_bytes = df_updated.to_csv(index=False).encode("utf-8")
    fileobj = io.BytesIO(csv_bytes)

    # 4) Commit the updated CSV
    msg = commit_message or f"append metrics row ({row.get('date', 'no-date')})"
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=branch,
        operations=[
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=fileobj),
        ],
        commit_message=msg,
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a JSON results file and append metrics to a CSV in a Hugging Face Space."
    )
    parser.add_argument("report", type=Path, help="Path to the JSON results file")
    parser.add_argument("--src-path", type=Path, default=".", help="Path to the source code (for cloc metrics)")
    parser.add_argument("--date", default=dt.date.isoformat(dt.datetime.now(tz=dt.UTC)), help="Date for the metrics row (default: today)")

    # NEW: Hub-related args
    parser.add_argument("--hub-repo", default=None, help="Space repo id, e.g. 'username/space-name'")
    parser.add_argument(
        "--hub-file", default="metrics.csv", help="Path of the CSV inside the repo (default: metrics.csv)"
    )
    parser.add_argument("--hub-branch", default="main", help="Branch to read/write (default: main)")
    parser.add_argument(
        "--hub-repo-type", default="space", choices=["space", "dataset", "model"], help="Repo type (default: space)"
    )
    parser.add_argument("--hub-token", default=None, help="HF token (otherwise taken from env/config)")

    args = parser.parse_args()

    df = load_report_to_df(args.report, only_leaves=True)
    df = process_df(df)
    agg = aggregate_metrics(df, date=args.date)
    agg.update(cloc_metrics(args.src_path))

    if not args.hub_repo:
        df_row = pd.DataFrame([agg])
        all_cols = list(dict.fromkeys(df_row.columns.tolist()))
        df_row = df_row.reindex(columns=all_cols)
        print(df_row.to_csv(index=False))
        sys.exit(0)

    sha = append_metrics_to_hub_csv(
        repo_id=args.hub_repo,
        path_in_repo=args.hub_file,
        row=agg,
        repo_type=args.hub_repo_type,
        branch=args.hub_branch,
        token=args.hub_token,
        commit_message=f"Append metrics for {agg.get('date', '')}",
    )
    print(f"Updated {args.hub_repo}/{args.hub_file} on branch {args.hub_branch}. Commit: {sha}")


if __name__ == "__main__":
    main()
