"""Dialect-trap taxonomy (paper Table I) and the residual-construct lint used by gate G2.

Each rule detects an ASE construct that should not survive into SQL Server code.
severity "error" fails G2; "warn" is recorded (deprecated but functional).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LintRule:
    rid: str
    category: str
    severity: str
    pattern: re.Pattern
    hint: str


def _r(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE | re.MULTILINE)


LINT_RULES: list[LintRule] = [
    LintRule("L01", "legacy_outer_join", "error", _r(r"\*=|=\*"),
             "Rewrite *= / =* as ANSI LEFT/RIGHT OUTER JOIN; keep inner-table predicates in ON."),
    LintRule("L02", "cursor_status", "error", _r(r"@@sqlstatus"),
             "Use @@FETCH_STATUS (0 = row fetched; -1/-2 otherwise)."),
    LintRule("L03", "cursor_status", "error", _r(r"\bdeallocate\s+cursor\b"),
             "SQL Server syntax is DEALLOCATE cursor_name."),
    LintRule("L04", "reporting_clause", "error", _r(r"\bcompute\s+(sum|avg|min|max|count)\s*\("),
             "COMPUTE BY is not supported; use GROUPING SETS/ROLLUP or separate queries."),
    LintRule("L05", "error_handling", "error", _r(r"\braiserror\s+\d+"),
             "Use RAISERROR('msg', severity, state) or THROW."),
    LintRule("L06", "quoting", "error", _r(r"\"[^\"\n]*[\s,:!%.][^\"\n]*\""),
             "Double-quoted text is an identifier under QUOTED_IDENTIFIER ON; use single quotes."),
    LintRule("L07", "messaging", "error", _r(r"%\d+!"),
             "ASE %n! placeholders are not expanded by SQL Server PRINT."),
    LintRule("L08", "null_semantics", "error", _r(r"\b(where|and|or|on|when|if|while|having)\b[^;\n]*?(?<![<>!])(=|<>|!=)\s*null\b"),
             "Use IS NULL / IS NOT NULL."),
    LintRule("L09", "locking_hints", "error", _r(r"\bfrom\s+\w+(?:\s+(?!with\b)\w+)?\s+(no)?holdlock\b|\bfrom\s+\w+(?:\s+(?!with\b)\w+)?\s+readpast\b"),
             "Use WITH (HOLDLOCK) / WITH (READPAST); noholdlock is read-committed (drop it, do not use NOLOCK)."),
    LintRule("L10", "locking_hints", "error", _r(r"\(\s*index\s+\w+\s*\)"),
             "Use WITH (INDEX(name))."),
    LintRule("L11", "builtin_functions", "error", _r(r"\b(str_replace|inttohex|hextoint|bintostr|strtobin)\s*\("),
             "ASE-only built-in. str_replace -> REPLACE (a NULL replacement deletes matches in ASE: use ''); "
             "inttohex/hextoint -> CONVERT via varbinary(4) with style 2 / style 1."),
    LintRule("L12", "datatypes", "error", _r(r"\bunsigned\s+(int|bigint|smallint|tinyint)\b"),
             "No unsigned types; widen (e.g. unsigned int -> bigint + CHECK >= 0)."),
    LintRule("L13", "datatypes", "error", _r(r"\bbig(date)?time\b"),
             "bigdatetime/bigtime -> datetime2(6)/time(6)."),
    LintRule("L14", "datatypes", "error", _r(r"\buni(var)?char\b|\bunitext\b"),
             "unichar/univarchar/unitext -> nchar/nvarchar/nvarchar(max)."),
    LintRule("L15", "ddl_storage", "error", _r(r"\block\s+(datarows|datapages|allpages)\b|\b(max_rows_per_page|exp_row_size|reservepagegap)\b"),
             "Remove ASE lock-scheme and page-storage options."),
    LintRule("L16", "transactions", "error", _r(r"@@tranchained|@@transtate|\bset\s+chained\b"),
             "Use @@OPTIONS & 2 / XACT_STATE() / SET IMPLICIT_TRANSACTIONS."),
    LintRule("L17", "datatypes", "warn", _r(r"\b\w+\s+(text|ntext|image)\b(?=\s*(null|not\s+null)?\s*[,)])"),
             "Deprecated LOB types; prefer varchar(max)/nvarchar(max)/varbinary(max)."),
    LintRule("L18", "datatypes", "warn", _r(r"\b\w+\s+timestamp\b(?=\s*(null|not\s+null)?\s*[,)])"),
             "timestamp is a deprecated synonym; use rowversion."),
]

CATEGORIES = {
    "legacy_outer_join": "Legacy *= / =* outer joins and inner-table restrictions",
    "cursor_status": "Cursor status globals and cursor statement syntax",
    "reporting_clause": "COMPUTE / COMPUTE BY report clauses",
    "error_handling": "RAISERROR argument forms and error-number ranges",
    "quoting": "Double-quoted string literals vs. QUOTED_IDENTIFIER",
    "messaging": "PRINT with %n! substitution arguments",
    "null_semantics": "= NULL comparisons and NULL in string concatenation",
    "grouping": "Transact-SQL GROUP BY extension (non-grouped select columns)",
    "locking_hints": "Unbracketed HOLDLOCK and (index name) hints",
    "builtin_functions": "ASE-only built-ins (str_replace, ...)",
    "datatypes": "Unsigned, bigdatetime, unichar/univarchar and LOB types",
    "ddl_storage": "Lock schemes and storage clauses in DDL",
    "transactions": "Chained mode and transaction-state globals",
    "string_semantics": "Empty-string-as-blank semantics",
    "control": "Portable code (measures harmful over-editing)",
    "composite": "Held-out multi-trap procedures (written after the checklist was frozen)",
}


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def lint(sql: str) -> list[dict]:
    text = _strip_comments(sql or "")
    findings = []
    for rule in LINT_RULES:
        for m in rule.pattern.finditer(text):
            findings.append({"rule": rule.rid, "category": rule.category, "severity": rule.severity,
                             "match": m.group(0)[:60], "hint": rule.hint})
    return findings


def trap_checklist() -> str:
    """Compact checklist injected into the trap-aware prompt."""
    lines = [f"- {r.rid} [{r.category}]: {r.hint}" for r in LINT_RULES]
    lines += [
        "- GROUP BY extension: ASE allows non-grouped columns in the select list and returns every "
        "row with its group aggregate; reproduce with a windowed aggregate (AGG() OVER (PARTITION BY ...)).",
        "- NULL concatenation: ASE treats NULL as '' in string concatenation; wrap nullable operands in ISNULL(x, '').",
        "- Outer-join restrictions: predicates on the inner table of an ASE *= join belong in the ON clause.",
    ]
    return "\n".join(lines)
