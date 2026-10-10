#!/usr/bin/env python3
"""Turn results/*.jsonl into CSV summaries, LaTeX tables, macros and figures for the paper.

  python scripts/analyze.py            # reads results/runs.jsonl (+ results/reconcile.jsonl if present)
  python scripts/analyze.py --corpus-only   # just regenerate the corpus table

Every number in paper/ comes from here. Do not hand-edit generated files in paper/tables/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verimig.corpus import load_corpus  # noqa: E402
from verimig.taxonomy import CATEGORIES  # noqa: E402

TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figures"
SHORT = {"rules": "Rules", "llm_plain": "Plain", "llm_trap": "Trap-aware", "agentic": "Agentic"}
CONV_LABEL = {"rules": "Rule baseline", "llm_plain": "LLM (plain)", "llm_trap": "LLM (trap-aware)",
              "agentic": "Agentic (G1/G2 repair)"}


def tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def case_bootstrap(per_case: list[float], iters: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    """Mean over cases with a 95% percentile bootstrap that resamples CASES (repetitions of the same
    case are highly correlated, so the case -- not the run -- is the independent unit)."""
    import random
    n = len(per_case)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(seed)
    means = sorted(sum(per_case[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(per_case) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def paired_effect(runs: list[dict], a: tuple, b: tuple, iters: int = 10000, seed: int = 11):
    """Difference in case-level G3 equivalence (b - a) with a paired 95% bootstrap over cases."""
    import random
    def per_case(cfg):
        acc = defaultdict(list)
        for r in runs:
            if config_name(r) == cfg and r["differential"]:
                acc[r["case_id"]].append(1.0 if r["g3"] else 0.0)
        return {k: sum(v) / len(v) for k, v in acc.items()}
    pa, pb = per_case(a), per_case(b)
    keys = sorted(pa.keys() & pb.keys())
    if not keys:
        return float("nan"), float("nan"), float("nan")
    d = [pb[k] - pa[k] for k in keys]
    rnd = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(d) / n, boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]


def go_split(sql: str) -> bool:
    """True if a GO separator cuts a CREATE PROCEDURE body (statements after GO that are not a new CREATE)."""
    import re
    parts = [p.strip() for p in re.split(r"^\s*go\s*;?\s*$", sql or "", flags=re.I | re.M)]
    parts = [p for p in parts if p]
    if len(parts) < 2 or not re.match(r"create\s+proc", parts[0], re.I):
        return False
    return any(not re.match(r"(create|alter|drop|grant|exec)\b", p, re.I) for p in parts[1:])


def pct(x: float) -> str:
    return "--" if x != x else f"{100 * x:.1f}"


def load_runs(path: Path) -> list[dict]:
    """Latest graded row per (case, converter, model, rep); split is taken from the corpus."""
    split = {c.id: c.split for c in load_corpus(ROOT / "corpus" / "cases.yaml")}
    latest = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "infra_error" in r:
            continue
        r["split"] = split.get(r["case_id"], "dev")
        latest[(r["case_id"], r["converter"], r["model"], r["rep"])] = r
    return list(latest.values())


def short_model(m: str) -> str:
    """Compact, paper-friendly model label, e.g. qwen2.5-coder:32b-instruct-q4_K_M -> Qwen2.5C-32B-Q4."""
    import re
    if m in ("-", "", None):
        return "--"
    fam = "Qwen2.5C" if m.startswith("qwen2.5-coder") else "DSCv2-Lite" if m.startswith("deepseek-coder-v2") \
        else m.split(":")[0]
    size = re.search(r"(\d+(?:\.\d+)?)b", m.split(":")[-1])
    q = re.search(r"q(\d)", m.split(":")[-1])
    out = fam + (f"-{size.group(1).upper()}B" if size else "")
    return out + (f"-Q{q.group(1)}" if q else "")


CONV_ORDER = {"rules": 0, "llm_plain": 1, "llm_trap": 2, "agentic": 3}


def _model_key(m: str):
    import re
    size = re.search(r"(\d+(?:\.\d+)?)b", m)
    return (0 if m.startswith("qwen") else 1, float(size.group(1)) if size else 0, m)


def config_name(r):
    return (r["converter"], r["model"])


def summarize(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[config_name(r)].append(r)
    rows = []
    for (conv, model), rs in sorted(groups.items(), key=lambda kv: (CONV_ORDER.get(kv[0][0], 9), _model_key(kv[0][1]))):
        diff = [r for r in rs if r["differential"]]
        ctrl = [r for r in diff if r["observability"] == "control"]
        g1_ok_diff = [r for r in diff if r["g1"]]
        silent = [r for r in diff if r["g1"] and r["g2"] and r["g3"] is False]
        per_case = defaultdict(list)
        for r in diff:
            per_case[r["case_id"]].append(1.0 if r["g3"] else 0.0)
        p, lo, hi = case_bootstrap([sum(v) / len(v) for v in per_case.values()])
        tps = [r["gen_tokens_per_s"] for r in rs if r.get("gen_tokens_per_s")]
        rows.append({
            "converter": conv, "model": model, "n_runs": len(rs), "n_diff": len(diff),
            "g1": sum(r["g1"] for r in rs) / len(rs),
            "g2": sum(r["g2"] for r in rs) / len(rs),
            "equiv": p, "equiv_lo": lo, "equiv_hi": hi,
            "full_pass": sum(1 for r in diff if r["g1"] and r["g2"] and r["g3"]) / max(1, len(diff)),
            "silent_given_g1": (sum(1 for r in g1_ok_diff if r["g3"] is False) / len(g1_ok_diff)) if g1_ok_diff else float("nan"),
            "silent_pass_g1g2": len(silent) / max(1, len(diff)),
            "control_preserved": (sum(1 for r in ctrl if r["g3"]) / len(ctrl)) if ctrl else float("nan"),
            "attempts": sum(r.get("attempts", 1) for r in rs) / len(rs),
            "latency_s": sum(r.get("latency_s", 0) for r in rs) / len(rs),
            "tok_s": (sum(tps) / len(tps)) if tps else float("nan"),
        })
    return rows


def write_csv(rows: list[dict], path: Path):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def table_main(rows: list[dict]):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Gate pass rates per configuration (\%). Equivalence is G3 over differential cases with a 95\% case-level bootstrap interval; "
             r"\emph{Silent} = passes G1 and G2 but fails G3; \emph{Ctrl} = portable control cases left behaviourally intact.}",
             r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Converter & Model & G1 & G2 & G3 equiv. [95\% CI] & Silent & Ctrl & Attempts & Latency (s) & Tok/s \\", r"\midrule"]
    for r in rows:
        tps_s = "--" if r["tok_s"] != r["tok_s"] else f"{r['tok_s']:.1f}"
        lines.append(
            f"{tex(CONV_LABEL.get(r['converter'], r['converter']))} & {tex(short_model(r['model']))} & {pct(r['g1'])} & {pct(r['g2'])} & "
            f"{pct(r['equiv'])} [{pct(r['equiv_lo'])}, {pct(r['equiv_hi'])}] & {pct(r['silent_pass_g1g2'])} & "
            f"{pct(r['control_preserved'])} & {r['attempts']:.2f} & {r['latency_s']:.1f} & "
            f"{tps_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES / "main_results.tex").write_text("\n".join(lines) + "\n")


def table_category(runs: list[dict], configs: list[tuple]):
    cats = sorted({r["category"] for r in runs if r["differential"]})
    cell = defaultdict(lambda: [0, 0])
    for r in runs:
        if r["differential"] and config_name(r) in configs:
            c = cell[(r["category"], config_name(r))]
            c[0] += bool(r["g3"])
            c[1] += 1
    head = " & ".join(tex(CONV_LABEL.get(c, c)) for c, _ in configs)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{G3 equivalence (\%) by trap category for the strongest model.}", r"\label{tab:category}",
             r"\footnotesize", r"\begin{tabular}{l" + "r" * len(configs) + "}", r"\toprule",
             f"Category & {head} \\\\", r"\midrule"]
    for cat in cats:
        vals = []
        for cfg in configs:
            k, n = cell[(cat, cfg)]
            vals.append(pct(k / n) if n else "--")
        lines.append(f"{tex(cat)} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "by_category.tex").write_text("\n".join(lines) + "\n")
    return cats, cell


def figure_category(cats, cell, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    w = 0.8 / max(1, len(configs))
    for i, cfg in enumerate(configs):
        ys = []
        for cat in cats:
            k, n = cell[(cat, cfg)]
            ys.append(100 * k / n if n else 0)
        ax.bar([x + i * w for x in range(len(cats))], ys, w, label=CONV_LABEL.get(cfg[0], cfg[0]))
    ax.set_xticks([x + w * (len(configs) - 1) / 2 for x in range(len(cats))])
    case_counts = Counter(c.category for c in load_corpus(ROOT / "corpus" / "cases.yaml") if c.differential)
    ax.set_xticklabels([f"{c} ({case_counts.get(c, 0)})" for c in cats], rotation=40, ha="right", fontsize=5.5)
    ax.set_ylabel("G3 equivalence (%)", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=5.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "category_g3.pdf")
    plt.close(fig)


def table_observability(runs: list[dict], configs: list[tuple]):
    order = [("compile", "Compile-visible"), ("silent", "Silent"), ("control", "Control")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{G3 equivalence (\%) by how an unconverted trap manifests, for the strongest model.}",
             r"\label{tab:observability}", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * len(configs) + "}", r"\toprule",
             "Trap class & " + " & ".join(tex(SHORT.get(c, c)) for c, _ in configs) + r" \\", r"\midrule"]
    out = {}
    for key, label in order:
        vals = []
        for cfg in configs:
            rs = [r for r in runs if r["differential"] and r["observability"] == key and config_name(r) == cfg]
            v = sum(bool(r["g3"]) for r in rs) / len(rs) if rs else float("nan")
            out[(key, cfg[0])] = v
            vals.append(pct(v))
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "observability.tex").write_text("\n".join(lines) + "\n")
    return out


def table_failures(runs: list[dict], configs: list[tuple], top: int = 8):
    """Cases most often failing silently (pass G1+G2, fail G3) for the given configs."""
    cases = {c.id: c for c in load_corpus(ROOT / "corpus" / "cases.yaml")}
    cnt = defaultdict(lambda: Counter())
    tot = defaultdict(lambda: Counter())
    kind = defaultdict(Counter)
    for r in runs:
        cfg = config_name(r)
        if cfg not in configs or not r["differential"]:
            continue
        tot[r["case_id"]][cfg[0]] += 1
        if r["g1"] and r["g2"] and r["g3"] is False:
            cnt[r["case_id"]][cfg[0]] += 1
            d = (r.get("g3_detail") or "")
            kind[r["case_id"]]["error" if ": error differs" in d else "rows" if ": rs differs" in d
                              else "messages" if ": messages differs" in d else "--"] += 1
    ranked = sorted(cnt, key=lambda k: -sum(cnt[k].values()))[:top]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Most frequent silent failures (passed G1 and G2, failed G3) for the strongest model: "
             r"failing runs / runs per converter, and the dominant divergence (rows, error, or messages).}",
             r"\label{tab:failures}", r"\footnotesize",
             r"\begin{tabular}{llrrrl}", r"\toprule",
             r"Case & Category & Plain & Trap & Agent & Diverges in \\", r"\midrule"]
    for k in ranked:
        c = cases.get(k)
        cells = [f"{cnt[k][cv]}/{tot[k][cv]}" if tot[k][cv] else "--" for cv in ("llm_plain", "llm_trap", "agentic")]
        dom = kind[k].most_common(1)[0][0] if kind[k] else "--"
        lines.append(f"{k} & {tex(c.category if c else '?')} & " + " & ".join(cells) + f" & {dom} \\\\")
    if not ranked:
        lines.append(r"\multicolumn{6}{c}{no silent failures observed} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "failures.tex").write_text("\n".join(lines) + "\n")


def _majority(runs: list[dict], cfg: tuple) -> dict:
    acc = defaultdict(list)
    for r in runs:
        if config_name(r) == cfg and r["differential"]:
            acc[r["case_id"]].append(bool(r["g3"]))
    return {k: sum(v) * 2 > len(v) for k, v in acc.items()}


def mcnemar(runs: list[dict], a: tuple, b: tuple) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on per-CASE majority G3 outcomes of configs a and b."""
    ra, rb = _majority(runs, a), _majority(runs, b)
    keys = ra.keys() & rb.keys()
    n01 = sum(1 for k in keys if ra[k] and not rb[k])
    n10 = sum(1 for k in keys if rb[k] and not ra[k])
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * tail)


def rep_agreement(runs: list[dict]) -> float:
    g = defaultdict(set)
    for r in runs:
        if r["converter"] != "rules" and r["differential"]:
            g[(r["converter"], r["model"], r["case_id"])].add(bool(r["g3"]))
    return sum(1 for v in g.values() if len(v) == 1) / len(g) if g else float("nan")


def go_share(runs: list[dict]) -> str:
    """Share of plain-prompt deployment failures caused by a GO splitting a procedure body."""
    f = [r for r in runs if r["converter"] == "llm_plain" and not r["g1"]]
    return pct(sum(go_split(r.get("candidate_sql", "")) for r in f) / len(f)) if f else "--"


def best_silent_share(runs: list[dict], cfg: tuple) -> str:
    """Of the best agent's G3 failures, the share that passed G1 and G2 (i.e., were silent)."""
    f = [r for r in runs if config_name(r) == cfg and r["differential"] and not r["g3"]]
    return pct(sum(1 for r in f if r["g1"] and r["g2"]) / len(f)) if f else "--"


def figure_scaling(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for conv, marker in (("llm_plain", "o"), ("llm_trap", "s"), ("agentic", "^")):
        pts = [(r["latency_s"], 100 * r["equiv"], r["model"]) for r in rows if r["converter"] == conv]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, s=18, label=CONV_LABEL[conv])
        if conv == "agentic":
            # alternate labels above/below in order of score so near-identical points stay readable
            for i, (x, y, m) in enumerate(sorted(pts, key=lambda p: (-p[1], p[0]))):
                ax.annotate(short_model(m), (x, y), fontsize=5, xytext=(5, (-8, 3, 3, 3, -8)[i % 5]),
                            textcoords="offset points")
            ax.set_xlim(right=max(p[0] for p in pts) * 4)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Mean latency per case (s, log)", fontsize=6.5)
    ax.set_ylabel("G3 equivalence (%)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "scaling.pdf")
    plt.close(fig)


def table_reconcile(path: Path) -> dict:
    agg = defaultdict(lambda: Counter())
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            a = agg[r["fault"]]
            a["n"] += 1
            for d in ("rowcount", "aggregate", "rowhash"):
                a[d] += bool(r[d])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{G4 detection rate (\%) for a single injected fault; \emph{clean} rows give the false-positive rate.}",
             r"\label{tab:reconcile}", r"\footnotesize", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Fault & Row count & Aggregate checksum & Row hash \\", r"\midrule"]
    for fault, a in agg.items():
        lines.append(f"{tex(fault)} & {pct(a['rowcount'] / a['n'])} & {pct(a['aggregate'] / a['n'])} & {pct(a['rowhash'] / a['n'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "reconcile.tex").write_text("\n".join(lines) + "\n")
    faulty = [f for f in agg if f != "clean"]
    def rate(d):
        n = sum(agg[f]["n"] for f in faulty)
        return sum(agg[f][d] for f in faulty) / n if n else float("nan")
    return {"rowcount": rate("rowcount"), "aggregate": rate("aggregate"), "rowhash": rate("rowhash")}


def table_corpus():
    cases = load_corpus(ROOT / "corpus" / "cases.yaml")
    by = defaultdict(list)
    for c in cases:
        by[c.category].append(c)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Dialect-trap taxonomy and pilot corpus. Obs.\ = how an unconverted construct manifests "
             r"(C = compile error, S = silent behaviour change, D = deprecated, K = control, M = mixed, held out).}",
             r"\label{tab:taxonomy}", r"\footnotesize", r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.66\columnwidth}cc}", r"\toprule",
             r"Category & Cases & Obs. \\", r"\midrule"]
    code = {"compile": "C", "silent": "S", "deprecation": "D", "control": "K", "composite": "M"}
    for cat, cs in by.items():
        obs = "/".join(sorted({code[c.observability] for c in cs}))
        lines.append(f"{tex(CATEGORIES.get(cat, cat))} & {len(cs)} & {obs} \\\\")
    lines += [r"\midrule", f"Total & {len(cases)} & \\\\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "corpus.tex").write_text("\n".join(lines) + "\n")
    return cases


def write_macros(values: dict):
    out = ["% Generated by scripts/analyze.py -- do not edit by hand."]
    for k, v in values.items():
        out.append(f"\\renewcommand{{\\{k}}}{{{v}}}")
    (TABLES / "macros.tex").write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--runs", default="results/runs.jsonl", help="run log to analyse")
    args = ap.parse_args()

    cases = table_corpus()
    macros = {"NumCases": str(len(cases)),
              "NumDiffCases": str(sum(1 for c in cases if c.differential)),
              "NumCategories": str(len({c.category for c in cases})),
              "NumSilentCases": str(sum(1 for c in cases if c.observability == "silent")),
              "NumControlCases": str(sum(1 for c in cases if c.observability == "control")),
              "NumReviewCases": str(sum(1 for c in cases if c.review)),
              "NumPolicyCases": str(sum(1 for c in cases if not c.differential)),
              "NumHeldout": str(sum(1 for c in cases if c.split == "heldout")),
              "NumSingleCall": str(sum(1 for c in cases if len(c.calls) == 1 and c.differential)),
              "NumDevCases": str(sum(1 for c in cases if c.split == "dev")),
              "MedianSrcLines": str(sorted(len(c.source.strip().splitlines()) for c in cases)[len(cases) // 2])}
    if args.corpus_only:
        write_macros(macros)
        print("corpus table + macros written")
        return

    all_runs = load_runs(ROOT / args.runs)
    if not all_runs:
        sys.exit(f"no graded runs in {args.runs} -- run the experiment first")
    runs = [r for r in all_runs if r["split"] == "dev"]          # main results: development corpus
    heldout = [r for r in all_runs if r["split"] == "heldout"]   # reported separately
    rows = summarize(runs)
    write_csv(rows, ROOT / "results" / "summary_by_config.csv")
    table_main(rows)

    llm_rows = [r for r in rows if r["converter"] == "agentic"]
    # best = highest agentic equivalence; ties broken by lower latency
    best_model = max(llm_rows, key=lambda r: (round(r["equiv"], 6), -r["latency_s"]))["model"] if llm_rows else "-"
    configs = [("rules", "-")] + [(c, best_model) for c in ("llm_plain", "llm_trap", "agentic")]
    configs = [c for c in configs if any(config_name(r) == c for r in runs)]
    cats, cell = table_category(runs, configs)
    figure_category(cats, cell, configs)
    obs = table_observability(runs, configs)
    table_failures(runs, configs)
    figure_scaling(rows)
    n01, n10, pval = mcnemar(runs, ("llm_plain", best_model), ("agentic", best_model))
    # Pre-specified contrasts for the strongest model, Holm-adjusted (3 tests)
    contrasts = {"PlainTrap": (("llm_plain", best_model), ("llm_trap", best_model)),
                 "TrapAgent": (("llm_trap", best_model), ("agentic", best_model)),
                 "RulesAgent": (("rules", "-"), ("agentic", best_model))}
    res = {k: mcnemar(runs, a, b) for k, (a, b) in contrasts.items()}
    order = sorted(res, key=lambda k: res[k][2])
    holm, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, res[k][2] * (len(order) - i)))
        holm[k] = running
    fmtp = lambda v: "p<0.001" if v < 0.001 else f"p={v:.3f}"  # noqa: E731
    contrast_macros = {}
    for k, (a01, a10, _) in res.items():
        contrast_macros[f"{k}Disc"] = f"{a01}/{a10}"
        contrast_macros[f"{k}P"] = fmtp(holm[k])
    eff, elo, ehi = paired_effect(runs, ("llm_plain", best_model), ("llm_trap", best_model))
    reff, rlo, rhi = paired_effect(runs, ("llm_trap", best_model), ("agentic", best_model))
    pp = lambda v: "--" if v != v else f"{100 * v:+.1f}"  # noqa: E731
    effect_macros = {"CheckEffect": pp(eff), "CheckEffectCI": f"[{pp(elo)}, {pp(ehi)}]",
                     "RepairEffect": pp(reff), "RepairEffectCI": f"[{pp(rlo)}, {pp(rhi)}]"}
    heldout_macros = {}
    if heldout:
        hs = {(r["converter"], r["model"]): r for r in summarize(heldout)}
        hg = lambda c, k, m=best_model: pct(hs[(c, m)][k]) if (c, m) in hs else "--"  # noqa: E731
        hag = [r for r in heldout if r["converter"] == "agentic" and r["model"] == best_model]
        heldout_macros = {
            "HeldoutRulesEquiv": hg("rules", "equiv", "-"), "HeldoutPlainEquiv": hg("llm_plain", "equiv"),
            "HeldoutTrapEquiv": hg("llm_trap", "equiv"), "HeldoutAgenticEquiv": hg("agentic", "equiv"),
            "HeldoutAgenticGOne": hg("agentic", "g1"), "HeldoutTrapGOne": hg("llm_trap", "g1"),
            "HeldoutRepairShare": pct(sum(r.get("attempts", 1) > 1 for r in hag) / len(hag)) if hag else "--",
            "HeldoutRuns": str(len(heldout)),
        }
    ag = [r for r in runs if r["converter"] == "agentic"]
    contrast_macros["RepairShare"] = pct(sum(r.get("attempts", 1) > 1 for r in ag) / len(ag)) if ag else "--"
    if heldout:
        write_csv(summarize(heldout), ROOT / "results" / "summary_heldout.csv")

    get = {(r["converter"], r["model"]): r for r in rows}
    def g(conv, key, model=best_model):
        r = get.get((conv, model))
        return pct(r[key]) if r else "--"
    models = sorted({r["model"] for r in rows if r["model"] != "-"})
    macros.update({
        "NumModels": str(len(models)),
        "NumRuns": str(len(runs)),
        "BestModel": tex(short_model(best_model)),
        "RulesEquiv": g("rules", "equiv", "-"),
        "PlainEquiv": g("llm_plain", "equiv"),
        "TrapEquiv": g("llm_trap", "equiv"),
        "AgenticEquiv": g("agentic", "equiv"),
        "PlainGOne": g("llm_plain", "g1"),
        "AgenticGOne": g("agentic", "g1"),
        "PlainSilent": g("llm_plain", "silent_pass_g1g2"),
        "AgenticSilent": g("agentic", "silent_pass_g1g2"),
        "PlainSilentGivenCompile": g("llm_plain", "silent_given_g1"),
        "PlainSilentClassEquiv": pct(obs.get(("silent", "llm_plain"), float("nan"))),
        "AgenticSilentClassEquiv": pct(obs.get(("silent", "agentic"), float("nan"))),
        "PlainCompileClassEquiv": pct(obs.get(("compile", "llm_plain"), float("nan"))),
        "AgenticCompileClassEquiv": pct(obs.get(("compile", "agentic"), float("nan"))),
        "McNemarDisc": f"{n01}/{n10}",
        "RepAgreement": pct(rep_agreement(runs)),
        **contrast_macros,
        **effect_macros,
        **heldout_macros,
        "GoSplitShare": go_share(runs),
        "BestAgentSilentShare": best_silent_share(runs, ("agentic", best_model)),
        "McNemarP": "p<0.001" if pval < 0.001 else f"p={pval:.3f}",  # used inside $...$
    })
    rec = ROOT / "results" / "reconcile.jsonl"
    if rec.exists():
        rr = table_reconcile(rec)
        macros.update({"RecRowcount": pct(rr["rowcount"]), "RecAggregate": pct(rr["aggregate"]),
                       "RecRowhash": pct(rr["rowhash"])})
    write_macros(macros)
    print("wrote results/summary_by_config.csv, paper/tables/*.tex, paper/figures/category_g3.pdf")


if __name__ == "__main__":
    main()
