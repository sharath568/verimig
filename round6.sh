#!/usr/bin/env bash
# Mutation follow-up: new boundary inputs for 8 cases, re-score those cases, re-run the mutation study.
# No LLM calls. ~20-30 min.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "Project folder: $PWD"
[ -f results/runs.jsonl ] || { echo "results/runs.jsonl not found -- wrong folder"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker not running: open -a Docker; docker compose up -d"; exit 1; }
$PY -m pytest -q
$PY scripts/check_env.py || true   # model check may fail if Ollama is off; references are what matter
[ -f results/mutation_initial.jsonl ] || cp results/mutation.jsonl results/mutation_initial.jsonl
$PY scripts/regrade.py --cases C02,C15,C38,C46,C47,C49,C50,C52
$PY scripts/mutation_study.py
$PY scripts/analyze.py
echo; echo "Done. Send back: results/runs.jsonl, results/mutation.jsonl"
