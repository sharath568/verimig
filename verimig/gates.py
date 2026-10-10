"""Verification gates.

G1  compile/deploy  : candidate deploys on SQL Server after the case setup.
G2  residual lint   : no error-severity ASE constructs remain (taxonomy.LINT_RULES).
G3  differential    : candidate and golden produce observationally equivalent outcomes for
                      every call (result sets, error presence + text, optional messages).
G4  reconciliation  : data-level checks, see verimig/reconcile.py.

Only G1/G2 feedback is ever shown to the agentic converter. G3 uses the golden oracle and is
kept out of the repair loop so that measured equivalence is not tuned against the oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Case
from .sqlutil import ExecOutcome, SqlServer
from .taxonomy import lint


@dataclass
class GateResult:
    g1: bool = False
    g1_error: str | None = None
    g2: bool = False
    g2_findings: list = field(default_factory=list)
    g3: bool | None = None  # None = not applicable (differential: false) or not reached
    g3_detail: str | None = None
    candidate_outcomes: list = field(default_factory=list)


def g2_lint(sql: str) -> tuple[bool, list]:
    findings = lint(sql)
    return not any(f["severity"] == "error" for f in findings), findings


def _comparable(o: ExecOutcome, case: Case):
    rs = [list(r) for r in o.resultsets]
    if not case.ordered:
        rs = [sorted(r, key=repr) for r in rs]
    err = (o.error[1].rstrip(".").strip() if o.error else None)
    msgs = o.messages if case.compare_messages else None
    return {"rs": rs, "error": err, "messages": msgs}


def compare_outcomes(ref: list[ExecOutcome], cand: list[ExecOutcome], case: Case) -> tuple[bool, str | None]:
    for i, (r, c) in enumerate(zip(ref, cand)):
        a, b = _comparable(r, case), _comparable(c, case)
        if a != b:
            for key in ("error", "rs", "messages"):
                if a[key] != b[key]:
                    return False, f"call {i}: {key} differs; expected={str(a[key])[:200]} got={str(b[key])[:200]}"
    return True, None


def run_calls(db: SqlServer, conn, case: Case) -> list[ExecOutcome]:
    return [db.execute_capture(conn, call) for call in case.calls]


def reference_outcomes(db: SqlServer, case: Case) -> list[ExecOutcome]:
    with db.scratch_db("vmref") as conn:
        ok, err = db.run_script(conn, case.setup)
        if not ok:
            raise RuntimeError(f"{case.id}: setup failed: {err}")
        ok, err = db.run_script(conn, case.golden)
        if not ok:
            raise RuntimeError(f"{case.id}: GOLDEN translation failed to deploy: {err}")
        return run_calls(db, conn, case)


def g1_compile_only(db: SqlServer, case: Case, candidate_sql: str) -> tuple[bool, str | None]:
    with db.scratch_db("vmg1") as conn:
        ok, err = db.run_script(conn, case.setup)
        if not ok:
            raise RuntimeError(f"{case.id}: setup failed: {err}")
        return db.run_script(conn, candidate_sql)


def evaluate(db: SqlServer, case: Case, candidate_sql: str, ref: list[ExecOutcome] | None) -> GateResult:
    res = GateResult()
    res.g2, res.g2_findings = g2_lint(candidate_sql)
    with db.scratch_db("vmcand") as conn:
        ok, err = db.run_script(conn, case.setup)
        if not ok:
            raise RuntimeError(f"{case.id}: setup failed: {err}")
        res.g1, res.g1_error = db.run_script(conn, candidate_sql)
        if not res.g1:
            res.g3 = False if case.differential else None
            return res
        outcomes = run_calls(db, conn, case)
        res.candidate_outcomes = [o.to_dict() for o in outcomes]
    if case.differential and ref is not None:
        res.g3, res.g3_detail = compare_outcomes(ref, outcomes, case)
    return res
