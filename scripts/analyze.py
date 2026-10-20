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
SHORT = {"rules": "Rules", "llm_plain": "Plain", "llm_trap": "Checklist", "agentic": "Repair loop"}
CONV_LABEL = {"rules": "Rule baseline", "llm_plain": "Plain prompt", "llm_trap": "Checklist prompt",
              "agentic": "Repair loop"}


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
    if m.startswith("anthropic:"):
        parts = m.split(":", 1)[1].split("-")          # claude-sonnet-5-5 -> Claude Sonnet 5.5
        return f"{parts[0].title()} {parts[1].title()} {'.'.join(parts[2:])} (hosted)"
    fam = "Qwen2.5C" if m.startswith("qwen2.5-coder") else "DSCv2-Lite" if m.startswith("deepseek-coder-v2") \
        else m.split(":")[0]
    size = re.search(r"(\d+(?:\.\d+)?)b", m.split(":")[-1])
    q = re.search(r"q(\d)", m.split(":")[-1])
    out = fam + (f"-{size.group(1).upper()}B" if size else "")
    return out + (f"-Q{q.group(1)}" if q else "")


CONV_ORDER = {"rules": 0, "llm_plain": 1, "llm_trap": 2, "agentic": 3}


def _model_key(m: str):
    import re
    if m.startswith("anthropic:"):
        return (2, 0, m)
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
             r"\caption{Gate pass rates per configuration (\%). G3 is agreement with the reference over differential cases with a 95\% case-level bootstrap interval; "
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
             r"\caption{Agreement with the references (\%) by category for the strongest model.}", r"\label{tab:category}",
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
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    case_counts = Counter(c.category for c in load_corpus(ROOT / "corpus" / "cases.yaml")
                          if c.differential and c.split == "dev")
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    w = 0.8 / max(1, len(configs))
    for i, cfg in enumerate(configs):
        ys = []
        for cat in cats:
            k, n = cell[(cat, cfg)]
            ys.append(100 * k / n if n else 0)
        ax.bar([x + i * w for x in range(len(cats))], ys, w, label=SHORT.get(cfg[0], cfg[0]))
    ax.set_xticks([x + w * (len(configs) - 1) / 2 for x in range(len(cats))])
    ax.set_xticklabels([f"{c.replace('_', ' ')} ({case_counts.get(c, 0)})" for c in cats],
                       rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Agreement (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.0, columnspacing=0.8)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "category_g3.pdf")
    plt.close(fig)


def table_observability(runs: list[dict], configs: list[tuple]):
    order = [("compile", "Compile-visible"), ("silent", "Silent"), ("control", "Control")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Agreement with the references (\%) by category type, for the strongest model.}",
             r"\label{tab:observability}", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * len(configs) + "}", r"\toprule",
             "Category type & " + " & ".join(tex(SHORT.get(c, c)) for c, _ in configs) + r" \\", r"\midrule"]
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
             r"\caption{Cases that most often passed G1 and G2 but failed G3, for the strongest model: failing runs "
             r"out of runs per converter, and where the first difference appeared.}",
             r"\label{tab:failures}", r"\footnotesize",
             r"\begin{tabular}{llrrrl}", r"\toprule",
             r"Case & Category & Plain & Checklist & Repair & Differs in \\", r"\midrule"]
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
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for conv, marker in (("llm_plain", "o"), ("llm_trap", "s"), ("agentic", "^")):
        pts = [(r["latency_s"], 100 * r["equiv"], r["model"]) for r in rows if r["converter"] == conv]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, s=22, label=SHORT[conv])
    # one label per model, beside its agentic point, pushed apart vertically so labels never overlap
    ag = sorted([(r["latency_s"], 100 * r["equiv"], r["model"]) for r in rows if r["converter"] == "agentic"],
                key=lambda p: -p[1])
    last_y = None
    for x, y, m in ag:
        ty = y if last_y is None or last_y - y >= 7 else last_y - 7
        ax.annotate(short_model(m).replace("Qwen2.5C-", "Qwen ").replace("DSCv2-Lite-", "DeepSeek "),
                    (x, y), xytext=(x * 1.5, ty), fontsize=7, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
        last_y = ty
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ag:
        ax.set_xlim(right=max(p[0] for p in ag) * 6)
    ax.set_xlabel("Mean model latency per case (s, log scale)", fontsize=8)
    ax.set_ylabel("Agreement (%)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "scaling.pdf")
    plt.close(fig)


ADVERSARIAL = {"null_to_literal", "delimiter_shift", "long_value_tail", "key_collision"}
DETECTORS = [("rowcount", "Count"), ("aggregate", "Aggr."),
             ("rowhash_naive", "Naive"), ("rowhash", "Unamb.")]


def table_reconcile(path: Path) -> dict:
    agg = defaultdict(lambda: Counter())
    ms = defaultdict(list)
    rows_changed = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            a = agg[r["fault"]]
            a["n"] += 1
            rows_changed[r["fault"]] = r.get("rows_changed", 0 if r["fault"] == "clean" else 1)
            for d, _ in DETECTORS:
                if d in r:
                    a[d] += bool(r[d])
                    ms[d].append(r.get("ms", {}).get(d, float("nan")))
    have = [(d, lab) for d, lab in DETECTORS if any(d in agg[f] or agg[f][d] for f in agg)]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{G4 detection rate (\%) per fault type over $n$ trials each. \emph{Rows}: target rows the "
             r"fault changes; \emph{clean} gives false positives. Detectors: row count, aggregate checksum, naive "
             r"row hash, unambiguous row hash. The last four faults are adversarial cases for the naive encoding.}",
             r"\label{tab:reconcile}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{lrr" + "r" * len(have) + "}", r"\toprule",
             "Fault & $n$ & Rows & " + " & ".join(lab for _, lab in have) + r" \\",
             r"\midrule"]
    order = [f for f in agg if f not in ADVERSARIAL] + [f for f in agg if f in ADVERSARIAL]
    for i, fault in enumerate(order):
        a = agg[fault]
        if i and fault in ADVERSARIAL and order[i - 1] not in ADVERSARIAL:
            lines.append(r"\midrule")
        cells = [pct(a[d] / a["n"]) for d, _ in have]
        lines.append(f"{tex(fault)} & {a['n']} & {rows_changed.get(fault, 1)} & " + " & ".join(cells) + r" \\")
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    lines.append(r"\midrule")
    lines.append("Median ms & & & " + " & ".join(f"{med(ms[d]):.0f}" for d, _ in have) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "reconcile.tex").write_text("\n".join(lines) + "\n")

    def rate(d, faults):
        k = sum(agg[f][d] for f in faults)
        n = sum(agg[f]["n"] for f in faults)
        p, lo, hi = wilson(k, n)
        return p, lo, hi, k, n
    std = [f for f in agg if f != "clean" and f not in ADVERSARIAL]
    adv = [f for f in agg if f in ADVERSARIAL]
    res = {}
    for d, _ in have:
        p, lo, hi, k, n = rate(d, std)
        res[d] = (p, lo, hi, k, n)
        if adv:
            res[d + "_adv"] = rate(d, adv)
    res["_ms"] = {d: med(ms[d]) for d, _ in have}
    res["_trials"] = agg[next(iter(agg))]["n"] if agg else 0
    res["_fp"] = sum(agg["clean"][d] for d, _ in have) if "clean" in agg else 0
    return res


def table_corpus():
    cases = load_corpus(ROOT / "corpus" / "cases.yaml")
    by = defaultdict(list)
    for c in cases:
        by[c.category].append(c)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Taxonomy of dialect differences and corpus. Obs.\ = how an unconverted construct shows up "
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


def env_macros() -> dict:
    """Hardware / container / engine details recorded by round3.sh in results/env_info.txt."""
    f = ROOT / "results" / "env_info.txt"
    if not f.exists():
        return {}
    kv = {}
    for line in f.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip().lower()] = v.strip().replace("b'", "").replace("'", "")
    out = {}
    if kv.get("chip"):
        out["HwChip"] = tex(kv["chip"])
    if kv.get("memory_bytes", "").isdigit():
        out["HwMemory"] = f"{int(kv['memory_bytes']) // 2**30}~GB"
    if kv.get("total number of cores"):
        out["HwCores"] = tex(kv["total number of cores"])
    if kv.get("macos"):
        out["HwMacOS"] = tex(kv["macos"])
    dig = kv.get("image_repo_digest", "")
    if "@sha256:" in dig:
        out["ImageDigest"] = "sha256:" + dig.split("@sha256:")[1][:12]
    if kv.get("sqlserver_build"):
        out["SqlBuild"] = tex(" ".join(kv["sqlserver_build"].split()[:2]))  # e.g. "16.0.4295.3 CU27"
    if kv.get("docker_version"):
        out["DockerVersion"] = tex(kv["docker_version"])
    return out


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
              "NumCategories": str(len({c.category for c in cases if c.category != "composite"})),
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

    llm_rows = [r for r in rows if r["converter"] == "agentic" and not r["model"].startswith("anthropic:")]
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
            "HeldoutTrapCI": f"[{hg('llm_trap', 'equiv_lo')}, {hg('llm_trap', 'equiv_hi')}]",
            "HeldoutAgenticCI": f"[{hg('agentic', 'equiv_lo')}, {hg('agentic', 'equiv_hi')}]",
            "HeldoutPlainCI": f"[{hg('llm_plain', 'equiv_lo')}, {hg('llm_plain', 'equiv_hi')}]",
        }
        weak = "qwen2.5-coder:7b"
        if ("agentic", weak) in hs and ("llm_trap", weak) in hs:
            heldout_macros.update({
                "WeakModel": tex(short_model(weak)),
                "WeakTrapGOne": hg("llm_trap", "g1", weak), "WeakAgenticGOne": hg("agentic", "g1", weak),
                "WeakTrapEquiv": hg("llm_trap", "equiv", weak), "WeakAgenticEquiv": hg("agentic", "equiv", weak),
                "WeakAttempts": f"{hs[('agentic', weak)]['attempts']:.1f}",
            })
    # checklist effect for EVERY model (the "best" model is chosen from results, so also report all)
    models_all = sorted({r["model"] for r in runs if r["converter"] == "llm_plain"
                         and not r["model"].startswith("anthropic:")})
    per_model = {m: mcnemar(runs, ("llm_plain", m), ("llm_trap", m)) for m in models_all}
    order_m = sorted(per_model, key=lambda m: per_model[m][2])
    holm_m, run_m = {}, 0.0
    for i, m in enumerate(order_m):
        run_m = max(run_m, min(1.0, per_model[m][2] * (len(order_m) - i)))
        holm_m[m] = run_m
    eff_m = {m: paired_effect(runs, ("llm_plain", m), ("llm_trap", m))[0] for m in models_all}
    contrast_macros["CheckPositiveModels"] = f"{sum(1 for v in eff_m.values() if v > 0)} of {len(models_all)}"
    contrast_macros["CheckSigModels"] = f"{sum(1 for v in holm_m.values() if v < 0.05)} of {len(models_all)}"
    contrast_macros["CheckEffectMin"] = pp(min(eff_m.values())) if eff_m else "--"
    contrast_macros["CheckEffectMax"] = pp(max(eff_m.values())) if eff_m else "--"
    # repair transitions: runs where the loop fired (first translation failed G1 or G2)
    rep = [r for r in runs if r["converter"] == "agentic" and r.get("attempts", 1) > 1 and r["differential"]]
    to_pass = sum(1 for r in rep if r["g3"])
    to_silent = sum(1 for r in rep if r["g1"] and r["g2"] and not r["g3"])
    still = len(rep) - to_pass - to_silent
    contrast_macros.update({"RepairRuns": str(len(rep)), "RepairToPass": str(to_pass),
                            "RepairToSilent": str(to_silent), "RepairStillLoud": str(still)})
    bf = [r for r in runs if config_name(r) == ("agentic", best_model) and r["differential"]
          and r["g1"] and r["g2"] and not r["g3"]]
    contrast_macros["BestSilentFirstAttempt"] = pct(sum(1 for r in bf if r.get("attempts", 1) == 1) / len(bf)) if bf else "--"
    ag = [r for r in runs if r["converter"] == "agentic"]
    contrast_macros["RepairShare"] = pct(sum(r.get("attempts", 1) > 1 for r in ag) / len(ag)) if ag else "--"
    if heldout:
        write_csv(summarize(heldout), ROOT / "results" / "summary_heldout.csv")

    get = {(r["converter"], r["model"]): r for r in rows}
    def g(conv, key, model=best_model):
        r = get.get((conv, model))
        return pct(r[key]) if r else "--"
    models = sorted({r["model"] for r in rows if r["model"] != "-" and not r["model"].startswith("anthropic:")})
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
        ci = lambda t: f"[{pct(t[1])}, {pct(t[2])}]"  # noqa: E731
        for d, m in (("rowcount", "Rowcount"), ("aggregate", "Aggregate"), ("rowhash", "Rowhash"),
                     ("rowhash_naive", "RowhashNaive")):
            if d in rr:
                macros[f"Rec{m}"] = pct(rr[d][0])
                macros[f"Rec{m}CI"] = ci(rr[d])
                macros[f"Rec{m}N"] = f"{rr[d][3]}/{rr[d][4]}"
            if d + "_adv" in rr:
                macros[f"Rec{m}Adv"] = pct(rr[d + "_adv"][0])
        macros["RecTrials"] = str(rr["_trials"])
        macros["RecMsRowhash"] = f"{rr['_ms'].get('rowhash', float('nan')):.0f}"
        macros["RecMsNaive"] = f"{rr['_ms'].get('rowhash_naive', float('nan')):.0f}"
        macros["RecFalsePos"] = str(rr["_fp"])
    # ---- hosted reference model (reported separately; never part of local-model claims)
    hosted = sorted({r["model"] for r in rows if r["model"].startswith("anthropic:")})
    if hosted:
        hm = hosted[0]
        hrow = {(r["converter"]): r for r in rows if r["model"] == hm}
        macros["HostedModel"] = tex(hm.split(":", 1)[1])
        for conv, key in (("llm_plain", "Plain"), ("llm_trap", "Trap"), ("agentic", "Agentic")):
            if conv in hrow:
                macros[f"Hosted{key}Equiv"] = pct(hrow[conv]["equiv"])
                macros[f"Hosted{key}CI"] = f"[{pct(hrow[conv]['equiv_lo'])}, {pct(hrow[conv]['equiv_hi'])}]"
                macros[f"Hosted{key}Silent"] = pct(hrow[conv]["silent_pass_g1g2"])
        if heldout:
            hh = {r["converter"]: r for r in summarize(heldout) if r["model"] == hm}
            if "agentic" in hh:
                macros["HostedHeldoutAgentic"] = pct(hh["agentic"]["equiv"])
            if "llm_plain" in hh:
                macros["HostedHeldoutPlain"] = pct(hh["llm_plain"]["equiv"])
    if hosted and "HostedAgenticEquiv" in macros:
        (TABLES / "hosted_setup.tex").write_text(
            "As a reference point we also ran one hosted model, " + macros["HostedModel"].replace("-", " ") +
            ", through its API with the same prompts and converters. The corpus is synthetic, so no confidential "
            "code was sent. The API has no seed parameter, and the hosted model is excluded from model selection "
            "and from the per-model tests.\n")
        (TABLES / "hosted_results.tex").write_text(
            "\\textit{Hosted reference model.} The hosted model reaches \\HostedPlainEquiv\\% with the plain prompt "
            "(95\\% CI \\HostedPlainCI), \\HostedTrapEquiv\\% with the checklist (\\HostedTrapCI), and "
            "\\HostedAgenticEquiv\\% with the repair loop (\\HostedAgenticCI); \\HostedPlainSilent\\% of its "
            "plain-prompt runs pass G1 and G2 but fail G3. On the held-out procedures it reaches "
            "\\HostedHeldoutPlain\\% and \\HostedHeldoutAgentic\\%.\n")
    else:
        (TABLES / "hosted_setup.tex").write_text(
            "Hosted models were not evaluated, because the setting we study is code that cannot leave the "
            "organisation.\n")
        (TABLES / "hosted_results.tex").write_text("")
    # ---- interface differences among translations that pass G3 (local models)
    local_all = [r for r in runs if not r["model"].startswith("anthropic:") and "iface_diff" in r]
    any_diff = [r for r in local_all if r["iface_diff"]]
    macros["IfaceAnyN"] = str(len(any_diff))
    macros["IfaceFailN"] = str(sum(1 for r in any_diff if r["g3"] is False))
    macros["IfacePolicyN"] = str(sum(1 for r in any_diff if r["g3"] is None))
    passed = [r for r in runs if r["differential"] and r["g3"] and not r["model"].startswith("anthropic:")
              and "iface_diff" in r]
    if passed:
        with_diff = [r for r in passed if r["iface_diff"]]
        macros["IfaceDiffShare"] = pct(len(with_diff) / len(passed))
        macros["IfaceDiffN"] = f"{len(with_diff)}/{len(passed)}"
        kinds = Counter(d.split(":")[-1].strip() for r in with_diff for d in r["iface_diff"])
        macros["IfaceTopKind"] = tex(kinds.most_common(1)[0][0]) if kinds else "--"
        macros["IfaceTypeShare"] = pct(sum(1 for r in with_diff if any("types differ" in d for d in r["iface_diff"]))
                                       / len(passed))
    # ---- mutation study
    mut = ROOT / "results" / "mutation.jsonl"
    if mut.exists():
        ms_ = [json.loads(l) for l in mut.read_text().splitlines() if l.strip()]
        ok = [m for m in ms_ if m["killed"] is not None]
        cls = {}
        cf = ROOT / "corpus" / "mutant_classification.yaml"
        if cf.exists():
            import yaml as _yaml
            cdata = _yaml.safe_load(cf.read_text()) or {}
            for kind in ("equivalent", "harness_limitation"):
                for e in cdata.get(kind) or []:
                    cls[(e["case"], e["operator"])] = (kind, e["reason"])
        killed = [m for m in ok if m["killed"]]
        surv = [m for m in ok if not m["killed"]]
        init = ROOT / "results" / "mutation_initial.jsonl"
        if init.exists():
            mi = [json.loads(l) for l in init.read_text().splitlines() if l.strip()]
            macros["MutKilledInitial"] = str(sum(1 for m in mi if m["killed"]))
            macros["MutTotalInitial"] = str(sum(1 for m in mi if m["killed"] is not None))
        equiv = [m for m in surv if cls.get((m["case_id"], m["operator"]), ("",))[0] == "equivalent"]
        limit = [m for m in surv if cls.get((m["case_id"], m["operator"]), ("",))[0] == "harness_limitation"]
        other = [m for m in surv if (m["case_id"], m["operator"]) not in cls]
        nonequiv = len(ok) - len(equiv)
        macros.update({"MutTotal": str(len(ok)), "MutKilled": str(len(killed)),
                       "MutScore": pct(len(killed) / len(ok)) if ok else "--",
                       "MutEquiv": str(len(equiv)), "MutLimit": str(len(limit)), "MutOther": str(len(other)),
                       "MutNonEquiv": str(nonequiv),
                       "MutScoreNonEquiv": pct(len(killed) / nonequiv) if nonequiv else "--",
                       "MutSurvived": str(len(surv))})
        groups = Counter(cls[(m["case_id"], m["operator"])][1] for m in equiv + limit)
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Mutants of the reference translations that the inputs did not detect, by reason.}",
                 r"\label{tab:mutants}", r"\footnotesize", r"\begin{tabular}{p{0.62\columnwidth}r}", r"\toprule",
                 r"Reason & Mutants \\", r"\midrule"]
        for reason, n in groups.most_common():
            kind = "equivalent" if any(cls[(m["case_id"], m["operator"])][1] == reason and
                                       cls[(m["case_id"], m["operator"])][0] == "equivalent" for m in equiv) else "G3 limitation"
            lines.append(f"{tex(reason[0].upper() + reason[1:])} ({kind}) & {n} \\\\")
        if other:
            lines.append(f"Not yet classified ({', '.join(sorted({m['case_id'] for m in other}))}) & {len(other)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TABLES / "mutants.tex").write_text("\n".join(lines) + "\n")
    macros.update(env_macros())
    write_macros(macros)
    print("wrote results/summary_by_config.csv, paper/tables/*.tex, paper/figures/category_g3.pdf")


if __name__ == "__main__":
    main()
