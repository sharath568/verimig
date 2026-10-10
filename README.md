# VeriMig — verification-gated LLM migration from Sybase ASE to SQL Server

VeriMig is a research artifact for the paper *"VeriMig: A Verification-Gated Agentic LLM Framework for Sybase ASE to SQL Server Migration"* (Kampili & Manda).

It translates SAP ASE (Sybase) T-SQL to SQL Server using **local LLMs**. Each translation is graded by four gates:

| Gate | Checks | Feeds the repair loop? |
|---|---|---|
| G1 deploy | The candidate deploys in a fresh SQL Server 2022 database | yes (error text) |
| G2 residual lint | No ASE-only constructs remain (`verimig/taxonomy.py`) | yes (findings) |
| G3 differential | Candidate and reviewed reference translation return identical result sets, errors, and messages | **no** — oracle is kept out of the loop |
| G4 reconcile | Row count vs. aggregate checksum vs. per-row SHA-256 on migrated data | n/a (separate study) |

The paper builds on Manda (2026), *Database Modernization – Sybase to SQL Server Migration* (IJMRSET 9(2), DOI 10.15680/IJMRSET.2026.0902049), and cites it as prior work. No text, tables, or figures from that article are reused. That article is © IJMRSET, and IEEE requires original work.

---

## 1. Step-by-step on your Mac

**a. Unzip and open a terminal in the folder**
```bash
cd ~/Downloads && unzip verimig.zip && cd verimig
```

**b. Install the tools (one time)**
```bash
brew install --cask docker          # open Docker Desktop once, then Settings -> General ->
                                    # turn ON "Use Rosetta for x86_64/amd64 emulation on Apple Silicon"
brew install ollama
brew install --cask mactex-no-gui   # optional: builds the PDF locally (otherwise use Overleaf)
```

**c. Set the database password**
```bash
cp .env.example .env
open -e .env                        # set MSSQL_SA_PASSWORD (8+ chars, upper, lower, digit, symbol)
```

**d. Quick end-to-end run first (~20–40 min, one 7B model, 1 repetition)**
```bash
make quick                          # same as: CONFIG=config.quick.yaml ./run_all.sh
```
This runs tests, starts SQL Server, pulls the model, checks that all 46 golden translations deploy,
runs every converter, runs the reconciliation study, and regenerates the paper tables. The data from
this run is for checking the pipeline only. Don't put it in the paper.

**e. Full run for the paper (overnight)**
```bash
make all                            # same as: ./run_all.sh
```
The full run covers 46 cases × 4 converters × 5 models × 3 repetitions, about 2,400 LLM translations.
On a 64 GB M-series Mac, expect roughly 10–15 hours, mostly in the 32B q8 model. It's resumable, so if
the Mac sleeps or you stop it, run the same command again. You can also go one model at a time:
```bash
python3 scripts/run_experiment.py --models qwen2.5-coder:7b
```
Run `caffeinate -i make all` to keep the Mac awake.

**f. Look at the results**
```bash
open results/summary_by_config.csv
open paper/main.pdf                 # tables, figures and abstract numbers are filled in
```

### Troubleshooting
| Symptom | Fix |
|---|---|
| `externally-managed-environment` | fixed: scripts now install into `.venv`. For manual commands, run `source .venv/bin/activate` first |
| `unknown flag: --env-file` / no `docker compose` | Docker Compose isn't installed. Install Docker Desktop (`brew install --cask docker`) or `brew install docker-compose` |
| `pip install pymssql` fails | `brew install freetds openssl` then `pip install pymssql` again |
| SQL Server container exits immediately | the password is too weak, or Rosetta is off; check `docker logs verimig-mssql` |
| Port 1433 in use | another SQL Server is running; stop it or change the port in `docker-compose.yml` and `config.yaml` |
| `check_env` says a golden failed | fix that case in `corpus/cases.yaml` before running; a broken oracle invalidates G3 |
| `check_env` warns PRINT not captured | message comparison (C07, C24) is skipped; mention it in the paper or upgrade pymssql |
| Ollama very slow / swapping | close other apps; the 32B q8 model needs ~35 GB of free memory |
| Want SSMA as a baseline | export each object to `results/imported/ssma/<case_id>.sql` (Windows), set `imported_tools: [ssma]` |

## 2. What "good data" means here

The numbers are only as good as the corpus and the run hygiene:
- **Corpus:** 46 cases across 15 categories (11 silent, 5 controls). Calls exercise boundary inputs
  where relevant (NULLs, empty strings, missing keys, month-end dates). 11 cases are marked `review: true`. Both
  authors must confirm their ASE behaviour (see §4).
- **Held-out set:** after you freeze `verimig/taxonomy.py`, write 10–20 new cases with `split: heldout`.
  `analyze.py` writes a separate `results/summary_heldout.csv`. This answers the reviewer question
  "did you tune the checklist to your own test set?"
- **Repetitions and stats:** 3 seeds per configuration, 95% Wilson intervals, and an exact McNemar test
  on paired runs (plain vs. agentic).
- **Provenance:** every run logs the candidate SQL, gate results, and model digests
  (`results/run_meta_*.json`). Keep these files; they are your evidence if a reviewer asks.

## 3. How the numbers reach the paper

`paper/` contains **no hand-typed results** (see `paper/HOW_TO_EDIT.md` for the file layout). Every figure in the abstract and results section is a macro (e.g. `\AgenticEquiv`) that defaults to a red **[TBD]**. `scripts/analyze.py` overwrites these macros in `paper/tables/macros.tex` from your runs. Blue `[Note: …]` markers are author to-dos and must all be removed before submission.

## 4. Validating the corpus (do this before reporting anything)

`corpus/cases.yaml` is a **pilot** corpus. Both authors should go through each case:

1. Confirm the ASE behaviour. The 12 cases marked `review: true` depend on ASE-specific semantics: C02, C06, C09, C10, C19, C26, C27, C28, C30, C32, C33, and C34. Run them on a real ASE 16 instance if you have access; SAP ASE has no ARM build, so this means an x86 VM or a work machine. Otherwise, cite the specific section of the ASE reference manual.
2. Confirm the golden translation is the behaviour you would sign off on in a real migration.
3. **Keep growing the corpus.** 46 cases is respectable for a 6-page paper; 60–100 is stronger. Add anonymised patterns from procedures you have both migrated. Write a set of **held-out cases after freezing `taxonomy.py`**, so that trap-aware prompting is not evaluated only on cases its checklist was written from (see Threats to Validity).

Case format is documented at the top of `cases.yaml`. Use `observability: silent` for traps that compile but change behaviour; these are the paper's central point.

## 4b. Correcting a reference after a run

If an audit shows a golden translation was wrong, fix it in `corpus/cases.yaml`, then re-score the stored
translations with no LLM calls (a backup of the run log is made first):
```bash
.venv/bin/python scripts/regrade.py --cases C06
.venv/bin/python scripts/analyze.py
```

## 5. Layout

```
run_all.sh                   one-command runner (CONFIG=config.quick.yaml for the quick profile)
config.yaml / config.quick.yaml   full and quick experiment profiles
corpus/cases.yaml            46 trap cases + golden translations
verimig/taxonomy.py          trap categories, 18 G2 lint rules, trap checklist for prompts
verimig/gates.py             G1–G3
verimig/reconcile.py         G4 detectors
verimig/converters/          rules baseline, Ollama LLM, agentic repair loop, SSMA import
verimig/sqlutil.py           scratch databases, batch splitting, outcome capture
scripts/                     check_env, run_experiment, reconcile_experiment, analyze
paper/                       IEEEtran LaTeX: main.tex + sections/*.tex + refs.bib (see paper/HOW_TO_EDIT.md)
tests/                       offline tests
```

## 6. Pre-submission checklist

- [ ] Both authors agree on author order, affiliations, and emails (fill in the `\note{email}`)
- [ ] Corpus reviewed (§4); held-out cases added
- [ ] Full runs complete; `make analyze && make paper`; no red [TBD] or blue [Note] left
- [ ] Results paragraphs rewritten to match what the data actually shows
- [ ] Every `refs.bib` entry verified against the publisher page (authors, pages, DOI)
- [ ] Repo public (GitHub) + Zenodo DOI in "Artifact Availability"
- [ ] Page limit and template rules of the target conference checked; IEEE PDF eXpress if required
- [ ] Self-run a similarity check if available; IEEE runs CrossCheck/iThenticate on submission
