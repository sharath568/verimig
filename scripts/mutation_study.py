#!/usr/bin/env python3
"""Mutation study: how many small, known bugs injected into the reference translations do the
case inputs detect? Measures test adequacy of G3 (reviewers: "two to four inputs may be optimistic").

Each operator makes a one-place change to a reference; the mutant is deployed and checked against
the unmodified reference with the full G3 contract. A mutant is *killed* if G1 or G3 fails.
Survivors are either equivalent mutants (no observable difference is possible, e.g. dropping
HOLDLOCK in a single session) or signs of weak inputs; they are listed for manual review.

  python scripts/mutation_study.py            -> results/mutation.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_env_file  # noqa: E402
from verimig.corpus import load_corpus  # noqa: E402
from verimig.gates import evaluate, reference_outcomes  # noqa: E402
from verimig.sqlutil import SqlServer  # noqa: E402

I = re.IGNORECASE
OPERATORS: list[tuple[str, str, str]] = [
    ("is_null", r"\bIS NULL\b", "= NULL"),
    ("is_not_null", r"\bIS NOT NULL\b", "<> NULL"),
    ("left_to_inner", r"\bLEFT( OUTER)? JOIN\b", "INNER JOIN"),
    ("right_to_inner", r"\bRIGHT( OUTER)? JOIN\b", "INNER JOIN"),
    ("drop_isnull", r"ISNULL\(([^,()]+),\s*''\)", r"\1"),
    ("gt_to_ge", r"(?<![<>!=])>(?!=)", ">="),
    ("lt_to_le", r"(?<![<>!=])<(?![=>])", "<="),
    ("drop_partition", r"OVER \(PARTITION BY [^)]*\)", "OVER ()"),
    ("bigint_to_int", r"\bbigint\b", "int"),
    ("dec20_to_bigint", r"\bdecimal\(20,\s*0\)", "bigint"),
    ("dt2_6_to_0", r"\bdatetime2\(6\)", "datetime2(0)"),
    ("time6_to_0", r"\btime\(6\)", "time(0)"),
    ("nvarchar_to_varchar", r"\bnvarchar\b", "varchar"),
    ("error_text", r"RAISERROR\('", "RAISERROR('!"),
    ("print_text", r"PRINT '", "PRINT '!"),
    ("drop_commit", r"\bCOMMIT TRAN;?", ""),
    ("return_code", r"\bRETURN 1\b", "RETURN 0"),
    ("sum_to_max", r"\bSUM\(", "MAX("),
    ("avg_to_max", r"\bAVG\(", "MAX("),
    ("order_desc", r"ORDER BY ([\w.]+)(?!\s+DESC)", r"ORDER BY \1 DESC"),
    ("drop_holdlock", r"\s*WITH \(HOLDLOCK\)", ""),
    ("top_plus", r"TOP \((\d+)\)", lambda m: f"TOP ({int(m.group(1)) + 1})"),
    ("replace_target", r"REPLACE\(([^,]+),\s*'-',\s*'_'\)", r"REPLACE(\1, '-', '')"),
]


def mutants(sql: str):
    for name, pat, rep in OPERATORS:
        m = re.search(pat, sql, I)
        if m:
            new = sql[:m.start()] + (rep(m) if callable(rep) else m.expand(rep)) + sql[m.end():]
            if new != sql:
                yield name, new


def main():
    load_env_file()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    s = cfg["sqlserver"]
    db = SqlServer(s["host"], s["port"], s["user"], os.environ[s["password_env"]])
    out = ROOT / "results" / "mutation.jsonl"
    total = killed = 0
    with out.open("w") as fh:
        for c in load_corpus(ROOT / cfg["experiment"]["corpus"]):
            if not c.differential:
                continue
            ref = reference_outcomes(db, c)
            for op, sql in mutants(c.golden):
                try:
                    gr = evaluate(db, c, sql, ref)
                    dead = (not gr.g1) or gr.g3 is False
                    detail = gr.g1_error if not gr.g1 else gr.g3_detail
                except Exception as exc:  # noqa: BLE001
                    dead, detail = None, repr(exc)[:200]
                total += 1
                killed += bool(dead)
                fh.write(json.dumps({"case_id": c.id, "split": c.split, "operator": op, "killed": dead,
                                     "detail": (detail or "")[:240], "mutant_sql": sql}) + "\n")
            print(f"[mut] {c.id}: done", flush=True)
    print(f"mutants: {total}, killed: {killed} -> results/mutation.jsonl")


if __name__ == "__main__":
    main()
