"""Verification gates.

G1  deploy          : candidate deploys on SQL Server after the case setup.
G2  residual lint   : no error-severity ASE constructs remain (taxonomy.LINT_RULES).
G3  differential    : candidate and reference produce the same observable behaviour for every call.
G4  reconciliation  : data-level checks, see verimig/reconcile.py.

Behavioural contract checked by G3, per call:
  * every result set, in order (rows as multisets only when the case declares `ordered: false`),
    with exact values (no rounding; a case may declare an explicit numeric `tolerance`);
  * the procedure return code (single EXEC calls are wrapped to capture it);
  * whether an error was raised and its message text (error numbers are not compared, because a
    correct translation may legitimately use a different user-error number);
  * informational (PRINT) messages, when the case sets `compare_messages: true`;
  * transaction state right after the call: @@TRANCOUNT and XACT_STATE().
and once after all calls:
  * the final contents of every user table (side effects), excluding rowversion columns.
Column names and aliases are not compared.

Only G1/G2 feedback is ever shown to the agentic converter. G3 uses the reference oracle and is
kept out of the repair loop so that measured equivalence is not tuned against the oracle.
"""
from __future__ import annotations

import decimal
import re
from dataclasses import dataclass, field

from .corpus import Case
from .sqlutil import ExecOutcome, SqlServer
from .taxonomy import lint

_SINGLE_EXEC = re.compile(r"^\s*EXEC(?:UTE)?\s+(\w+)(.*)$", re.IGNORECASE | re.DOTALL)
TRAN_PROBE = "SELECT @@TRANCOUNT AS trancount, XACT_STATE() AS xact_state"


def wrap_call(call: str) -> str:
    """Capture the return code of a plain single-procedure call: EXEC p args -> EXEC @rc = p args."""
    m = _SINGLE_EXEC.match(call)
    if not m or ";" in call or "@" in m.group(1):
        return call
    return f"DECLARE @__rc int; EXEC @__rc = {m.group(1)}{m.group(2)}; SELECT @__rc AS __return_code"


@dataclass
class Reference:
    outcomes: list
    state: dict
    signatures: dict = field(default_factory=dict)


@dataclass
class GateResult:
    g1: bool = False
    g1_error: str | None = None
    g2: bool = False
    g2_findings: list = field(default_factory=list)
    g3: bool | None = None  # None = not applicable (differential: false)
    g3_detail: str | None = None
    iface_diff: list = field(default_factory=list)  # interface differences; reported, not gating
    candidate_outcomes: list = field(default_factory=list)


def interface_diff(ref: Reference, ref_outs: list, cand_sig: dict, cand_outs: list) -> list[str]:
    """Differences in procedure signatures and result-set column types. Not part of G3: a correct
    translation may legitimately widen a parameter type (e.g. unsigned int -> bigint)."""
    diffs = []
    for proc in sorted(set(ref.signatures) | set(cand_sig)):
        a, b = ref.signatures.get(proc), cand_sig.get(proc)
        if a is None or b is None:
            diffs.append(f"procedure {proc} missing in {'reference' if a is None else 'candidate'}")
        elif [x[0] for x in a] != [x[0] for x in b] or [x[-1] for x in a] != [x[-1] for x in b]:
            diffs.append(f"{proc}: parameter names/OUTPUT flags differ")
        elif a != b:
            diffs.append(f"{proc}: parameter types differ")
    for i, (r, c) in enumerate(zip(ref_outs, cand_outs)):
        if r.coltypes and c.coltypes and r.coltypes != c.coltypes:
            diffs.append(f"call {i}: result column types differ")
    return diffs


def g2_lint(sql: str) -> tuple[bool, list]:
    findings = lint(sql)
    return not any(f["severity"] == "error" for f in findings), findings


def _num_close(a, b, tol: float) -> bool:
    try:
        return abs(decimal.Decimal(a) - decimal.Decimal(b)) <= decimal.Decimal(str(tol))
    except (decimal.InvalidOperation, TypeError):
        return a == b


def _rows_equal(ra: list, rb: list, tol: float | None) -> bool:
    if tol is None:
        return ra == rb
    if len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        if len(x) != len(y) or not all(_num_close(p, q, tol) for p, q in zip(x, y)):
            return False
    return True


def _comparable(o: ExecOutcome, case: Case):
    rs = [list(r) for r in o.resultsets]
    if not case.ordered:
        rs = [sorted(r, key=repr) for r in rs]
    err = (o.error[1].rstrip(".").strip() if o.error else None)
    msgs = o.messages if case.compare_messages else None
    return {"rs": rs, "error": err, "messages": msgs, "tran": o.tran}


def compare_outcomes(ref: list[ExecOutcome], cand: list[ExecOutcome], case: Case) -> tuple[bool, str | None]:
    tol = case.extra.get("tolerance")
    for i, (r, c) in enumerate(zip(ref, cand)):
        a, b = _comparable(r, case), _comparable(c, case)
        for key in ("error", "rs", "messages", "tran"):
            same = (len(a["rs"]) == len(b["rs"]) and all(_rows_equal(x, y, tol) for x, y in zip(a["rs"], b["rs"]))) \
                if key == "rs" else a[key] == b[key]
            if not same:
                return False, f"call {i}: {key} differs; expected={str(a[key])[:200]} got={str(b[key])[:200]}"
    return True, None


def compare_state(ref: dict, cand: dict) -> tuple[bool, str | None]:
    for t in sorted(set(ref) | set(cand)):
        if ref.get(t) != cand.get(t):
            return False, f"final state: table {t} differs; expected={str(ref.get(t))[:160]} got={str(cand.get(t))[:160]}"
    return True, None


LOST = "connection lost (statement timeout or crash)"


def run_calls(db: SqlServer, conn, case: Case) -> tuple[list[ExecOutcome], int | None]:
    """Run every call; returns (outcomes, index of the call after which the connection died or None).

    A call that never finishes (e.g. a cursor loop that does not terminate) hits the statement
    timeout, after which the driver drops the connection. Remaining calls are then recorded as
    failed instead of crashing the run.
    """
    outs = []
    for i, call in enumerate(case.calls):
        o = db.execute_capture(conn, wrap_call(call))
        if not db.alive(conn):
            o.error = (-1, LOST)
            outs.append(o)
            outs += [ExecOutcome(error=(-1, LOST)) for _ in case.calls[i + 1:]]
            return outs, i
        t = db.execute_capture(conn, TRAN_PROBE)
        o.tran = t.resultsets[0][0] if t.resultsets and t.resultsets[0] else None
        outs.append(o)
    return outs, None


def reference_outcomes(db: SqlServer, case: Case) -> Reference:
    with db.scratch_db("vmref") as conn:
        ok, err = db.run_script(conn, case.setup)
        if not ok:
            raise RuntimeError(f"{case.id}: setup failed: {err}")
        ok, err = db.run_script(conn, case.golden)
        if not ok:
            raise RuntimeError(f"{case.id}: REFERENCE translation failed to deploy: {err}")
        sig = db.signatures(conn)
        outs, dead = run_calls(db, conn, case)
        if dead is not None:
            raise RuntimeError(f"{case.id}: REFERENCE lost its connection at call {dead} (timeout?)")
        return Reference(outs, db.snapshot(conn), sig)


def g1_compile_only(db: SqlServer, case: Case, candidate_sql: str) -> tuple[bool, str | None]:
    with db.scratch_db("vmg1") as conn:
        ok, err = db.run_script(conn, case.setup)
        if not ok:
            raise RuntimeError(f"{case.id}: setup failed: {err}")
        return db.run_script(conn, candidate_sql)


def evaluate(db: SqlServer, case: Case, candidate_sql: str, ref: Reference | None) -> GateResult:
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
        cand_sig = db.signatures(conn)
        outcomes, dead = run_calls(db, conn, case)
        state = db.snapshot(conn) if dead is None else None
        if ref is not None:
            res.iface_diff = interface_diff(ref, ref.outcomes, cand_sig, outcomes)
        res.candidate_outcomes = [o.to_dict() for o in outcomes]
    if case.differential and ref is not None and dead is not None:
        res.g3, res.g3_detail = False, f"call {dead}: error differs; expected=completion got={LOST}"
        return res
    if case.differential and ref is not None:
        res.g3, res.g3_detail = compare_outcomes(ref.outcomes, outcomes, case)
        if res.g3:
            res.g3, res.g3_detail = compare_state(ref.state, state)
    return res
