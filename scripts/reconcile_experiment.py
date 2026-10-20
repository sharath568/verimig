#!/usr/bin/env python3
"""Fault-injection study for gate G4 (data reconciliation).

Loads a seeded source table, then for each fault type and trial loads a target copy with one injected
fault and records which detectors notice it and how long each took. Most faults change one row;
swap_amounts changes two (recorded in the `rows_changed` field). 'clean' trials
(no fault) measure false positives. The adversarial faults null_to_literal, delimiter_shift and
long_value_tail target the ambiguity of the naive row encoding.

  python scripts/reconcile_experiment.py --rows 2000 --trials 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from scripts.run_experiment import load_env_file  # noqa: E402
from verimig.reconcile import detect  # noqa: E402
from verimig.sqlutil import SqlServer  # noqa: E402

DDL = """CREATE TABLE cust (
  id int NOT NULL, name nvarchar(60) NOT NULL, city varchar(40) NOT NULL,
  amt money NOT NULL, ts datetime2(6) NOT NULL, note varchar(max) NULL)"""
COLS = {"id": None, "name": None, "city": None, "amt": 2, "ts": 126, "note": None}
NAMES = ["Ana Souza", "José Álvarez", "Zoë Martin", "Ångström AB", "Müller GmbH", "Nguyễn Văn An",
         "Priya Raman", "Li Wei", "Ólafur Örn", "Chloé Dubois", "Sven Berg", "Amara Okafor",
         "Pipe|Works", "Bar|Tools"]
CITIES = ["Austin", "Dallas", "Hyderabad", "Berlin", "Lyon", "Oslo", "Lagos", "Hanoi"]
FAULTS = ["clean", "drop_row", "dup_row", "truncate_text", "charset_loss", "money_round",
          "ts_precision", "null_to_empty", "swap_amounts",
          # adversarial: target the ambiguity of the naive encoding, and duplicate keys
          "null_to_literal", "delimiter_shift", "long_value_tail", "key_collision"]
ROWS_CHANGED = {f: 1 for f in FAULTS} | {"clean": 0, "swap_amounts": 2}


def make_rows(n: int, seed: int) -> list[list]:
    rnd = random.Random(seed)
    base = dt.datetime(2025, 1, 1)
    rows = []
    for i in range(1, n + 1):
        rows.append([
            i,
            rnd.choice(NAMES) + f" {i}",
            rnd.choice(CITIES),
            decimal.Decimal(rnd.randint(0, 10_000_000)) / decimal.Decimal(10_000),
            base + dt.timedelta(seconds=rnd.randint(0, 30_000_000), microseconds=rnd.randint(1, 999_999)),
            None if rnd.random() < 0.3 else
            ("L" * 6000 + f"-{i}") if rnd.random() < 0.05 else f"note-{rnd.randint(0, 999)}",
        ])
    return rows


def inject(rows: list[list], fault: str, rnd: random.Random) -> list[list]:
    rows = [list(r) for r in rows]
    if fault == "clean":
        return rows
    if fault == "drop_row":
        rows.pop(rnd.randrange(len(rows)))
    elif fault == "dup_row":
        rows.append(list(rows[rnd.randrange(len(rows))]))
    elif fault == "truncate_text":
        r = rnd.choice([r for r in rows if len(r[1]) > 10])
        r[1] = r[1][:10]
    elif fault == "charset_loss":
        r = rnd.choice([r for r in rows if any(ord(ch) > 127 for ch in r[1])])
        r[1] = "".join(ch if ord(ch) < 128 else "?" for ch in r[1])
    elif fault == "money_round":
        r = rnd.choice([r for r in rows if r[3] != r[3].quantize(decimal.Decimal("0.01"))])
        r[3] = r[3].quantize(decimal.Decimal("0.01"))
    elif fault == "ts_precision":
        r = rnd.choice(rows)
        r[4] = r[4].replace(microsecond=(r[4].microsecond // 1000) * 1000)
    elif fault == "null_to_empty":
        r = rnd.choice([r for r in rows if r[5] is None])
        r[5] = ""
    elif fault == "null_to_literal":
        r = rnd.choice([r for r in rows if r[5] is None])
        r[5] = "<NULL>"
    elif fault == "delimiter_shift":
        r = rnd.choice([r for r in rows if "|" in r[1]])
        head, tail = r[1].split("|", 1)
        r[1], r[2] = head, tail + "|" + r[2]          # 'A|B','C' -> 'A','B|C'
    elif fault == "long_value_tail":
        r = rnd.choice([r for r in rows if r[5] and len(r[5]) > 4000])
        r[5] = r[5][:-1] + ("X" if r[5][-1] != "X" else "Y")
    elif fault == "key_collision":
        a, b = rnd.sample(rows, 2)
        b[0] = a[0]                                    # b now carries a's key; b's own key disappears
    elif fault == "swap_amounts":
        a, b = rnd.sample([r for r in rows], 2)
        while a[3] == b[3]:
            a, b = rnd.sample(rows, 2)
        a[3], b[3] = b[3], a[3]
    return rows


def _lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, decimal.Decimal):
        return format(v, "f")
    if isinstance(v, dt.datetime):
        return "'" + v.isoformat(sep=" ", timespec="microseconds") + "'"
    return "N'" + str(v).replace("'", "''") + "'"


def load(conn, rows, batch: int = 500):
    """Multi-row INSERTs with explicit literals (keeps microseconds and 4-dp money intact)."""
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE cust")
    for i in range(0, len(rows), batch):
        values = ",".join("(" + ",".join(_lit(v) for v in r) + ")" for r in rows[i:i + batch])
        cur.execute(f"INSERT cust (id, name, city, amt, ts, note) VALUES {values}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    load_env_file()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    s = cfg["sqlserver"]
    db = SqlServer(s["host"], s["port"], s["user"], os.environ[s["password_env"]])
    src_rows = make_rows(args.rows, args.seed)
    out = ROOT / "results" / "reconcile.jsonl"
    with db.scratch_db("vmsrc") as src, db.scratch_db("vmtgt") as tgt, out.open("w") as fh:
        for conn in (src, tgt):
            conn.cursor().execute(DDL)
        load(src, src_rows)
        for fault in FAULTS:
            for t in range(args.trials):
                rnd = random.Random(f"{args.seed}-{fault}-{t}")
                load(tgt, inject(src_rows, fault, rnd))
                res = detect(src, tgt, "cust", "id", COLS)
                fh.write(json.dumps({"fault": fault, "trial": t, "rows": args.rows,
                                     "rows_changed": ROWS_CHANGED[fault], **res}) + "\n")
            print(f"[g4] {fault}: done")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
