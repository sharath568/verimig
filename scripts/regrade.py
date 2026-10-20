#!/usr/bin/env python3
"""Re-score stored translations against the CURRENT corpus, with no LLM calls.

Use this after correcting a reference translation (golden) or a case's calls. The candidate SQL
stored in the run log is re-deployed and re-checked by G1-G3. Model outputs are not regenerated,
so this is safe: the agent never saw G3, and only the oracle changed.

  python scripts/regrade.py --cases C06                 # rewrites results/runs.jsonl in place
  python scripts/regrade.py --cases C06 --runs results/quick_runs.jsonl

A timestamped backup of the original log is written next to it first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_env_file  # noqa: E402
from verimig.corpus import load_corpus  # noqa: E402
from verimig.gates import evaluate, reference_outcomes  # noqa: E402
from verimig.sqlutil import SqlServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="comma-separated case ids to re-score")
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    load_env_file()
    cfg = yaml.safe_load(Path(args.config).read_text())
    s = cfg["sqlserver"]
    db = SqlServer(s["host"], s["port"], s["user"], os.environ[s["password_env"]])
    wanted = set(args.cases.split(","))
    cases = {c.id: c for c in load_corpus(ROOT / cfg["experiment"]["corpus"]) if c.id in wanted}
    missing = wanted - set(cases)
    if missing:
        sys.exit(f"unknown case ids: {sorted(missing)}")

    path = ROOT / args.runs
    backup = path.with_suffix(f".bak_{dt.datetime.now():%Y%m%dT%H%M%S}.jsonl")
    shutil.copy(path, backup)
    print(f"backup -> {backup.relative_to(ROOT)}")

    refs = {cid: reference_outcomes(db, c) for cid, c in cases.items()}
    out, changed, total, failed = [], 0, 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("case_id") in cases and "infra_error" not in r:
            total += 1
            if total % 100 == 0:
                print(f"  ... {total} rows re-scored", flush=True)
            c = cases[r["case_id"]]
            gr = None
            for attempt in range(2):  # one retry for transient container hiccups
                try:
                    gr = evaluate(db, c, r["candidate_sql"], refs[c.id])
                    break
                except Exception as exc:  # noqa: BLE001
                    err = repr(exc)
            if gr is None:
                failed += 1
                r["regrade_error"] = err[:300]
                out.append(json.dumps(r))
                print(f"[warn] {r['case_id']} {r['converter']} {r['model']} r{r['rep']}: {err[:120]}")
                continue
            r.pop("regrade_error", None)
            before = (r["g1"], r["g2"], r["g3"])
            r.update({"g1": gr.g1, "g1_error": gr.g1_error, "g2": gr.g2, "g3": gr.g3, "g3_detail": gr.g3_detail, "iface_diff": gr.iface_diff,
                      "lint_errors": [f["rule"] for f in gr.g2_findings if f["severity"] == "error"],
                      "lint_warnings": [f["rule"] for f in gr.g2_findings if f["severity"] == "warn"],
                      "regraded_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            if before != (r["g1"], r["g2"], r["g3"]):
                changed += 1
        out.append(json.dumps(r))
    path.write_text("\n".join(out) + "\n")
    print(f"re-scored {total} rows; {changed} changed outcome; {failed} could not be re-scored")
    if failed:
        print("Rows that could not be re-scored keep their old scores and carry a 'regrade_error' field.")


if __name__ == "__main__":
    main()
