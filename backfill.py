#!/usr/bin/env python3
"""
Backfill monthly snapshots of a Git repo and run analyze.py at each snapshot.

Example:
  python backfill.py \
      --repo /path/to/repo \
      --branch main \
      --src-subdir . \
      --analyzer-dir /path/to/code-checker \
      --report-name result.json
"""
import argparse
import datetime as dt
import io
import sys
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd


def run(cmd: list[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=check)


def git_current_ref(repo: Path) -> tuple[str, str]:
    ref = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
    sha = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    return ref, sha


def git_earliest_first_parent_sha(repo: Path, branch: str) -> Optional[str]:
    """Return the earliest commit SHA on the branch's first-parent history."""
    cp = run(["git", "rev-list", branch, "--first-parent", "--reverse"], cwd=repo, check=False)
    for line in cp.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def git_first_commit_date(repo: Path, branch: str) -> dt.date:
    """Return the date of the earliest first-parent commit on the branch."""
    sha = git_earliest_first_parent_sha(repo, branch)
    if not sha:
        raise RuntimeError(f"No commits found on branch {branch}")
    out = run(["git", "show", "-s", "--format=%cI", sha], cwd=repo).stdout.strip()
    return dt.datetime.fromisoformat(out.replace("Z", "+00:00")).date()


def git_first_commit_on_or_after(repo: Path, branch: str, since_iso: str) -> Optional[str]:
    cp = run(["git", "rev-list", branch, "--first-parent", f"--since={since_iso}", "--reverse"], cwd=repo, check=False)
    shas = [line.strip() for line in cp.stdout.splitlines() if line.strip()]
    return shas[0] if shas else None


def git_commit_date(repo: Path, sha: str) -> dt.datetime:
    out = run(["git", "show", "-s", "--format=%cI", sha], cwd=repo).stdout.strip()
    return dt.datetime.fromisoformat(out.replace("Z", "+00:00"))


def month_starts_descending(today: Optional[dt.date] = None) -> list[dt.date]:
    if today is None:
        today = dt.date.today()
    d = today.replace(day=1)
    starts: list[dt.date] = []
    # walk back up to 200 years (hard safety cap)
    for _ in range(2400):
        starts.append(d)
        year = d.year - 1 if d.month == 1 else d.year
        month = 12 if d.month == 1 else d.month - 1
        d = dt.date(year, month, 1)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill monthly snapshots of a Git repo and run analyze.py on each.")
    parser.add_argument("--repo", type=Path, required=True, help="Path to the target Git repository to analyze.")
    parser.add_argument("--branch", default="main", help="Branch name to traverse (default: main).")
    parser.add_argument("--src-subdir", default="src", help="Subdirectory inside repo to analyze (default: .).")
    parser.add_argument("--analyzer-dir", type=Path, default=Path("."), help="Directory containing main.py and analyze.py.")
    parser.add_argument("--report-name", default="result.json", help="Filename for JSON report (default: result.json).")
    parser.add_argument("--max-months", type=int, default=100, help="Optional limit on number of months to process.")
    args = parser.parse_args()

    repo = args.repo.resolve()
    analyzer_dir = args.analyzer_dir.resolve()
    src_dir = (repo / args.src_subdir).resolve()

    if not (analyzer_dir / "main.py").exists():
        print(f"error: main.py not found in {analyzer_dir}", file=sys.stderr)
        sys.exit(2)

    if not (Path(".") / "analyze.py").exists():
        print("error: analyze.py not found", file=sys.stderr)
        sys.exit(2)

    start_ref, start_sha = git_current_ref(repo)

    try:
        run(["git", "show-ref", "--verify", f"refs/heads/{args.branch}"], cwd=repo)
    except subprocess.CalledProcessError:
        print(f"error: branch '{args.branch}' not found in {repo}", file=sys.stderr)
        sys.exit(2)

    # ensure first-parent follows the chosen branch’s lineage
    run(["git", "checkout", "--quiet", args.branch], cwd=repo)

    earliest = git_first_commit_date(repo, args.branch)
    starts = month_starts_descending()
    if args.max_months:
        starts = starts[: args.max_months]

    dfs: list[pd.DataFrame] = []
    seen_shas: set[str] = set()

    try:
        for month_start in starts:
            # progress bar
            print("|", end="", flush=True, file=sys.stderr)
            if month_start < earliest.replace(day=1):
                break

            since_iso = dt.datetime(month_start.year, month_start.month, 1, tzinfo=dt.timezone.utc).isoformat()
            sha = git_first_commit_on_or_after(repo, args.branch, since_iso)
            if not sha or sha in seen_shas:
                continue
            seen_shas.add(sha)

            # checkout snapshot
            run(["git", "checkout", "--quiet", sha], cwd=repo)

            # run main.py to produce JSON report for this snapshot
            report_path = analyzer_dir / args.report_name
            cp_report = run([sys.executable, "main.py", str(src_dir), "-o", str(report_path)], cwd=analyzer_dir, check=False)
            if cp_report.returncode != 0:
                print(f"error: main.py failed on commit {sha} ({git_commit_date(repo, sha).date().isoformat()}), aborting", file=sys.stderr)
                print(cp_report.stderr, file=sys.stderr)
                break

            # run analyze.py (prints single-row CSV)
            commit_dt = git_commit_date(repo, sha).date().isoformat()
            cp_agg = run([sys.executable, "analyze.py", str(report_path), "--src-path", str(src_dir), "--date", commit_dt])
            csv_text = cp_agg.stdout.strip()
            if not csv_text:
                continue

            df = pd.read_csv(io.StringIO(csv_text))
            dfs.append(df)

        if not dfs:
            print("no data collected", file=sys.stderr)
            return

        out = pd.concat(dfs, ignore_index=True)
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.sort_values("date").reset_index(drop=True)
            out["date"] = out["date"].dt.date.astype(str)
        sys.stdout.write(out.to_csv(index=False))

    finally:
        # restore original state
        if start_ref == "HEAD":
            run(["git", "checkout", "--quiet", start_sha], cwd=repo, check=False)
        else:
            run(["git", "checkout", "--quiet", start_ref], cwd=repo, check=False)


if __name__ == "__main__":
    main()
