#!/usr/bin/env bash
# Camera-ready run (reviewer fixes). No LLM calls: stored translations are re-scored with the
# corrected G3 comparator, and the G4 reconciliation study is re-run with the new encoding.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31mERROR: %s\033[0m\n" "$*"; exit 1; }

echo "Project folder: $PWD"
[ -f results/runs.jsonl ] || die "results/runs.jsonl not found -- run this inside the project folder used for the experiments."
[ -x "$PY" ] || die ".venv missing -- run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"

step "1/6 Offline tests"
$PY -m pytest -q

step "2/6 Recording hardware, container and engine details -> results/env_info.txt"
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "chip: $(sysctl -n machdep.cpu.brand_string)"
  echo "cpu_cores_total: $(sysctl -n hw.ncpu)"
  echo "memory_bytes: $(sysctl -n hw.memsize)"
  system_profiler SPHardwareDataType 2>/dev/null | grep -E "Model Name|Model Identifier|Chip|Total Number of Cores|Memory" | sed 's/^ *//'
  echo "macos: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "docker_version: $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
  echo "container_platform_and_image: $(docker inspect verimig-mssql --format '{{.Platform}} {{.Image}}' 2>/dev/null)"
  echo "image_repo_digest: $(docker image inspect mcr.microsoft.com/mssql/server:2022-latest --format '{{index .RepoDigests 0}}' 2>/dev/null)"
  echo "ollama_version: $(ollama --version 2>/dev/null | tail -1)"
  $PY - <<'PY'
import os, yaml, sys
sys.path.insert(0, ".")
from scripts.run_experiment import load_env_file
from verimig.sqlutil import SqlServer
load_env_file()
s = yaml.safe_load(open("config.yaml"))["sqlserver"]
cur = SqlServer(s["host"], s["port"], s["user"], os.environ[s["password_env"]]).connect().cursor()
cur.execute("SELECT SERVERPROPERTY('ProductVersion'), SERVERPROPERTY('ProductUpdateLevel'), SERVERPROPERTY('Edition')")
print("sqlserver_build: %s %s %s" % tuple(cur.fetchone()))
PY
} | tee results/env_info.txt

step "3/6 Pre-flight: every reference deploys under the extended behavioural contract"
$PY scripts/check_env.py

step "4/6 Re-score ALL stored translations with the corrected comparator (no LLM calls, ~1-2 h)"
ALL=$($PY -c "import sys;sys.path.insert(0,'.');from verimig.corpus import load_corpus;print(','.join(c.id for c in load_corpus('corpus/cases.yaml')))")
$PY scripts/regrade.py --cases "$ALL"

step "5/6 G4 reconciliation study with the unambiguous encoding (~15-25 min)"
$PY scripts/reconcile_experiment.py --rows 2000 --trials 30

step "6/6 Tables, figures, macros"
$PY scripts/analyze.py
echo; echo "Done. Send back: results/runs.jsonl, results/reconcile.jsonl, results/env_info.txt"
