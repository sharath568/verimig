"""Offline tests (no SQL Server / Ollama needed):  python -m pytest -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verimig.converters import convert_rules  # noqa: E402
from verimig.corpus import load_corpus  # noqa: E402
from verimig.gates import compare_outcomes  # noqa: E402
from verimig.sqlutil import ExecOutcome, extract_sql, normalize_value, split_batches  # noqa: E402
from verimig.taxonomy import lint  # noqa: E402

CASES = load_corpus(ROOT / "corpus" / "cases.yaml")


def errors(sql):
    return [f for f in lint(sql) if f["severity"] == "error"]


def test_corpus_loads():
    assert len(CASES) >= 20
    assert all(c.calls for c in CASES)


def test_goldens_are_lint_clean():
    for c in CASES:
        assert not errors(c.golden), (c.id, errors(c.golden))


def test_lint_flags_sources_except_known_gaps():
    # Categories that regex lint cannot see by design (only G3 catches them).
    lint_blind = {"control", "grouping", "string_semantics", "datatypes"}
    for c in CASES:
        if c.category in lint_blind or c.id in {"C09", "C27", "C31"}:
            continue
        assert errors(c.source), c.id


def test_assignment_null_not_flagged():
    assert not errors("UPDATE t SET c = NULL WHERE id = 1")
    assert errors("SELECT id FROM t WHERE c = null")


def test_split_and_extract():
    assert split_batches("a\nGO\nb\ngo\n") == ["a", "b"]
    assert extract_sql("blah\n```sql\nSELECT 1;\n```\n") == "SELECT 1;"


def test_normalize_numbers():
    import decimal
    assert normalize_value(decimal.Decimal("100.0000")) == normalize_value(100)
    assert normalize_value(20) == "20"


def test_compare_detects_differences():
    c = CASES[0]
    a = ExecOutcome(resultsets=[[("1", "Ana")]])
    b = ExecOutcome(resultsets=[[("1", "Ann")]])
    assert compare_outcomes([a], [a], c)[0]
    assert not compare_outcomes([a], [b], c)[0]


def test_rules_baseline_reduces_lint_errors():
    before = sum(len(errors(c.source)) for c in CASES)
    after = sum(len(errors(convert_rules(c.source).sql)) for c in CASES)
    assert after < before
