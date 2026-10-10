#!/usr/bin/env python3
"""Run the conversion experiment: every (case x converter x model x repetition).

Results are appended to results/runs.jsonl; re-running resumes and skips finished rows.

  python scripts/run_experiment.py                      # full grid from config.yaml
  python scripts/run_experiment.py --cases C01,C08 --models qwen2.5-coder:7b --reps 1   # smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verimig.converters import (Ollama, convert_agentic, convert_imported,  # noqa: E402
                                convert_llm, convert_rules)
from verimig.corpus import load_corpus  # noqa: E402
from verimig.gates import evaluate, g1_compile_only, g2_lint, reference_outcomes  # noqa: E402
from verimig.sqlutil import SqlServer  # noqa: E402


def load_env_file():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "infra_error" in r:
                    continue  # retried on the next run
                keys.add((r["case_id"], r["converter"], r["model"], r["rep"]))
    return keys


def write_meta(cfg, ollama_url: str):
    meta = {"started_utc": dt.datetime.utcnow().isoformat(), "platform": platform.platform(),
            "machine": platform.machine(), "python": platform.python_version(), "models": {}}
    try:
        meta["ollama_version"] = requests.get(f"{ollama_url}/api/version", timeout=10).json().get("version")
        tags = requests.get(f"{ollama_url}/api/tags", timeout=10).json().get("models", [])
        for t in tags:
            if t["name"] in cfg["experiment"]["models"]:
                meta["models"][t["name"]] = {"digest": t.get("digest"), "size": t.get("size"),
                                             "details": t.get("details")}
    except Exception as exc:  # noqa: BLE001
        meta["ollama_error"] = str(exc)
    out = ROOT / "results" / f"run_meta_{dt.datetime.utcnow():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"[meta] {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--cases")
    ap.add_argument("--models")
    ap.add_argument("--converters")
    ap.add_argument("--reps", type=int)
    args = ap.parse_args()

    load_env_file()
    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    if args.models:
        exp["models"] = args.models.split(",")
    if args.converters:
        exp["converters"] = args.converters.split(",")
    reps = args.reps or exp["repetitions"]

    s = cfg["sqlserver"]
    pwd = os.environ.get(s["password_env"])
    if not pwd:
        sys.exit(f"Set {s['password_env']} (see .env.example)")
    db = SqlServer(s["host"], s["port"], s["user"], pwd)
    oc = cfg["ollama"]
    client = Ollama(oc["url"], oc["temperature"], oc["num_ctx"], oc["timeout_s"])

    cases = load_corpus(ROOT / exp["corpus"])
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]

    out_path = ROOT / exp["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out_path)
    write_meta(cfg, oc["url"])

    # Oracle outcomes once per case; a failing golden aborts the run (fix the corpus first).
    refs = {}
    for c in cases:
        refs[c.id] = reference_outcomes(db, c)
        print(f"[ref] {c.id} ok")

    jobs = []
    for conv in exp["converters"]:
        model_list = exp["models"] if conv.startswith("llm") or conv == "agentic" else ["-"]
        jobs += [(conv, m) for m in model_list]
    jobs += [(f"imported:{t}", "-") for t in exp.get("imported_tools", [])]

    with out_path.open("a") as fh:
        for conv, model in jobs:
            conv_reps = 1 if (conv == "rules" or conv.startswith("imported")) else reps
            for c in cases:
                for rep in range(conv_reps):
                    key = (c.id, conv, model, rep)
                    if key in finished:
                        continue
                    seed = exp["base_seed"] + rep * 100
                    try:
                        if conv == "rules":
                            cr = convert_rules(c.source)
                        elif conv.startswith("imported:"):
                            cr = convert_imported(c.id, ROOT / "results" / "imported" / conv.split(":", 1)[1])
                            if cr is None:
                                print(f"[skip] {conv} has no file for {c.id}")
                                continue
                        elif conv in ("llm_plain", "llm_trap"):
                            cr = convert_llm(client, model, c.source, seed, trap_aware=(conv == "llm_trap"))
                        elif conv == "agentic":
                            cr = convert_agentic(client, model, c.source, seed,
                                                 compile_fn=lambda sql, c=c: g1_compile_only(db, c, sql),
                                                 lint_fn=g2_lint, max_iters=exp["agentic_max_iters"])
                        else:
                            raise ValueError(conv)
                        gr = evaluate(db, c, cr.sql, refs[c.id])
                        row = {
                            "ts": dt.datetime.utcnow().isoformat(), "case_id": c.id, "category": c.category,
                            "observability": c.observability, "differential": c.differential, "split": c.split,
                            "converter": conv, "model": model, "rep": rep, "seed": seed,
                            "g1": gr.g1, "g1_error": gr.g1_error, "g2": gr.g2,
                            "lint_errors": [f["rule"] for f in gr.g2_findings if f["severity"] == "error"],
                            "lint_warnings": [f["rule"] for f in gr.g2_findings if f["severity"] == "warn"],
                            "g3": gr.g3, "g3_detail": gr.g3_detail,
                            "attempts": cr.attempts, "latency_s": round(cr.latency_s, 3),
                            "prompt_tokens": cr.prompt_tokens, "output_tokens": cr.output_tokens,
                            "gen_tokens_per_s": cr.gen_tokens_per_s,
                            "candidate_sql": cr.sql, "trace": cr.trace,
                        }
                    except Exception as exc:  # noqa: BLE001  -- infrastructure failure, record & continue
                        row = {"ts": dt.datetime.utcnow().isoformat(), "case_id": c.id, "category": c.category,
                               "observability": c.observability, "differential": c.differential,
                               "converter": conv, "model": model, "rep": rep, "seed": seed,
                               "infra_error": repr(exc)}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    status = "INFRA" if "infra_error" in row else \
                        f"G1={int(row['g1'])} G2={int(row['g2'])} G3={row['g3']}"
                    print(f"[{conv}|{model}|r{rep}] {c.id}: {status}")


if __name__ == "__main__":
    main()
