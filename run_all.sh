#!/usr/bin/env bash
# One-command runner for VeriMig on macOS.
#   ./run_all.sh                          full experiment (config.yaml) -- run overnight
#   CONFIG=config.quick.yaml ./run_all.sh quick end-to-end check (~20-40 min)
# Safe to re-run: finished translations are skipped.
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${CONFIG:-config.yaml}"
step() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31mERROR: %s\033[0m\n" "$*"; exit 1; }

step "1/8 Checking tools"
command -v docker  >/dev/null || die "Docker Desktop not found (brew install --cask docker)"
command -v ollama  >/dev/null || die "Ollama not found (brew install ollama)"
command -v python3 >/dev/null || die "python3 not found"
[ -f .env ] || die "No .env file. Run: cp .env.example .env  and set MSSQL_SA_PASSWORD"

step "2/8 Python dependencies (project virtualenv in .venv)"
# Homebrew Python blocks system-wide pip installs (PEP 668), so everything runs inside .venv.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || die "could not create .venv (try: brew install python@3.12)"
fi
PY="$PWD/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt || die "pip install failed. If pymssql failed: brew install freetds openssl, then re-run"

step "3/8 Offline tests"
"$PY" -m pytest -q

step "4/8 Starting SQL Server 2022 (Docker)"
docker info >/dev/null 2>&1 || die "Docker is installed but not running. Open Docker Desktop (or start Colima: colima start --arch x86_64 --memory 8) and re-run."
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  die "Docker Compose is missing. Easiest fix: brew install --cask docker (Docker Desktop includes it).
       Or: brew install docker-compose   (then re-run)"
fi
# compose reads .env from this folder automatically
$COMPOSE up -d
printf "waiting for SQL Server to become healthy"
for i in $(seq 1 60); do
  state=$(docker inspect -f '{{.State.Health.Status}}' verimig-mssql 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && { echo " ok"; break; }
  printf "."; sleep 5
  [ "$i" = 60 ] && die "SQL Server not healthy. Check: docker logs verimig-mssql"
done

step "5/8 Ollama models"
curl -sf http://localhost:11434/api/version >/dev/null || { (ollama serve >/tmp/ollama.log 2>&1 &); sleep 5; }
for m in $("$PY" -c "import yaml;print(' '.join(yaml.safe_load(open('$CONFIG'))['experiment']['models']))"); do
  ollama pull "$m"
done

step "6/8 Pre-flight (every golden deploys, models present)"
"$PY" scripts/check_env.py --config "$CONFIG"

step "7/8 Experiments"
"$PY" scripts/run_experiment.py --config "$CONFIG"
"$PY" scripts/reconcile_experiment.py --rows 2000 --trials 30

step "8/8 Analysis + paper"
RUNS=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['experiment']['output'])")
"$PY" scripts/analyze.py --runs "$RUNS"
if command -v latexmk >/dev/null; then
  (cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null) && echo "PDF: paper/main.pdf"
else
  echo "latexmk not installed -- upload the paper/ folder to Overleaf, or: brew install --cask mactex-no-gui"
fi
echo; echo "Done. Summary: results/summary_by_config.csv"
