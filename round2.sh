#!/usr/bin/env bash
# Round-2 revision run: re-score existing translations on the new boundary inputs (no LLM calls),
# then translate + grade the 6 held-out procedures with every model (LLM calls), then rebuild tables.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }

step "1/5 Offline tests"
$PY -m pytest -q

step "2/5 Pre-flight: every reference (incl. new inputs and held-out cases) deploys"
$PY scripts/check_env.py

step "3/5 Re-score stored translations on the new inputs (no LLM calls, ~30-60 min)"
DEV=$($PY -c "import sys;sys.path.insert(0,'.');from verimig.corpus import load_corpus;print(','.join(c.id for c in load_corpus('corpus/cases.yaml') if c.split=='dev' and c.differential))")
$PY scripts/regrade.py --cases "$DEV"

step "4/5 Held-out procedures: all converters x models x 3 reps (LLM calls, ~20-40 min)"
$PY scripts/run_experiment.py --cases C47,C48,C49,C50,C51,C52

step "5/5 Tables, figures, macros"
$PY scripts/analyze.py
echo; echo "Done. Send results/runs.jsonl back for the round-2 audit."
