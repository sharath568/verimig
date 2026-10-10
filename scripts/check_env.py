#!/usr/bin/env python3
"""Pre-flight: SQL Server reachable, every golden deploys, Ollama up, models pulled."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import load_env_file  # noqa: E402
from verimig.corpus import load_corpus  # noqa: E402
from verimig.gates import reference_outcomes  # noqa: E402
from verimig.sqlutil import SqlServer  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    load_env_file()
    cfg = yaml.safe_load(Path(args.config).read_text())
    s = cfg["sqlserver"]
    ok = True
    try:
        db = SqlServer(s["host"], s["port"], s["user"], os.environ[s["password_env"]])
        cur = db.connect().cursor()
        cur.execute("SELECT @@VERSION")
        print("[sqlserver]", cur.fetchone()[0].splitlines()[0])
        probe = db.execute_capture(db.connect(), "PRINT 'verimig-probe'")
        if "verimig-probe" not in probe.messages:
            ok = False
            print("[sqlserver] WARNING: PRINT messages are not being captured by this pymssql build; "
                  "case C07 (compare_messages) would be scored without message comparison.")
        for c in load_corpus(ROOT / cfg["experiment"]["corpus"]):
            outs = reference_outcomes(db, c)
            shown = [o.resultsets or o.error or o.messages for o in outs]
            print(f"  golden {c.id}: OK  -> {str(shown)[:110]}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("[sqlserver] FAILED:", exc)
    try:
        url = cfg["ollama"]["url"]
        have = {m["name"] for m in requests.get(f"{url}/api/tags", timeout=10).json()["models"]}
        for m in cfg["experiment"]["models"]:
            flag = "OK" if m in have else "MISSING -> ollama pull " + m
            ok &= m in have
            print(f"[ollama] {m}: {flag}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("[ollama] FAILED:", exc)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
