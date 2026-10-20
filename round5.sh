#!/usr/bin/env bash
# Final camera-ready run: re-score with interface metadata + corrected C13 input (no LLM calls),
# mutation study of the inputs (no LLM calls), and the hosted reference model (Anthropic API).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31mERROR: %s\033[0m\n" "$*"; exit 1; }
echo "Project folder: $PWD"
[ -f results/runs.jsonl ] || die "results/runs.jsonl not found -- run inside the project folder used for the experiments."
docker info >/dev/null 2>&1 || die "Docker is not running: open -a Docker, then docker compose up -d"

step "1/5 Offline tests"
$PY -m pytest -q

step "2/5 Pre-flight (references deploy)"
$PY scripts/check_env.py

step "3/5 Re-score all stored translations (C13 input fix + interface metadata; no LLM calls, ~1.5 h)"
ALL=$($PY -c "import sys;sys.path.insert(0,'.');from verimig.corpus import load_corpus;print(','.join(c.id for c in load_corpus('corpus/cases.yaml')))")
$PY scripts/regrade.py --cases "$ALL"

step "4/5 Mutation study of the case inputs (~10 min)"
$PY scripts/mutation_study.py

step "5/5 Hosted reference model"
set -a; . ./.env; set +a
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  $PY scripts/run_experiment.py --config config.hosted.yaml
else
  echo "ANTHROPIC_API_KEY not set in .env -- skipping the hosted model (the paper then reports it as not run)."
fi

$PY scripts/analyze.py
echo; echo "Done. Send back: results/runs.jsonl, results/mutation.jsonl"
