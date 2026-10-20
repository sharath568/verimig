"""Gate G4: data reconciliation between a source and a target table.

Detectors, from weakest to strongest:

  rowcount       COUNT(*) only (the most common check in practice).
  aggregate      COUNT + CHECKSUM_AGG(BINARY_CHECKSUM(*)): SQL Server's built-in table checksum.
                 BINARY_CHECKSUM computes a 32-bit checksum per row and CHECKSUM_AGG combines the row
                 checksums into one value; Microsoft does not document either algorithm in detail.
  rowhash_naive  per-row SHA-256 over values joined with '|' and NULL written as '<NULL>' (the
                 encoding used in the first version of VeriMig). It is AMBIGUOUS: NULL and the text
                 '<NULL>' collide, and ('A|B','C') collides with ('A','B|C'). Kept only as a baseline.
  rowhash        per-row SHA-256 over an unambiguous encoding: every column is written as
                 'N;' for NULL or 'V<length>:<value>;' otherwise, using nvarchar(max) so nothing is
                 truncated. Distinct rows always give distinct encodings, so a difference can only be
                 missed through a SHA-256 collision. Rows are compared as a multiset of (key, hash)
                 pairs, so duplicate and missing keys are detected too.

Value text uses CONVERT with an explicit style per column (126 for date/time types, 2 for money,
none otherwise) so that no precision is lost in the conversion.
"""
from __future__ import annotations

import time
from collections import Counter


def _conv(col: str, style: int | None) -> str:
    return f"CONVERT(nvarchar(max), {col}, {style})" if style is not None else f"CONVERT(nvarchar(max), {col})"


def _robust_expr(cols: dict[str, int | None]) -> str:
    parts = []
    for c, st in cols.items():
        v = _conv(c, st)
        parts.append(f"CASE WHEN {c} IS NULL THEN CAST(N'N;' AS nvarchar(max)) "
                     f"ELSE N'V' + CAST(DATALENGTH({v}) / 2 AS nvarchar(20)) + N':' + {v} + N';' END")
    return " + ".join(parts)


def _naive_expr(cols: dict[str, int | None]) -> str:
    return " + N'|' + ".join(
        f"ISNULL(CONVERT(nvarchar(4000), {c}{', ' + str(st) if st is not None else ''}), N'<NULL>')"
        for c, st in cols.items())


def rowcount(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT_BIG(*) FROM {table}")
    return cur.fetchone()[0]


def aggregate_sig(conn, table: str) -> tuple:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT_BIG(*), CHECKSUM_AGG(BINARY_CHECKSUM(*)) FROM {table}")
    return tuple(cur.fetchone())


def _hash_multiset(conn, table: str, key: str, expr: str) -> Counter:
    cur = conn.cursor()
    cur.execute(f"SELECT {key}, CONVERT(varchar(64), HASHBYTES('SHA2_256', {expr}), 2) FROM {table}")
    return Counter((k, h) for k, h in cur.fetchall())


def detect(src_conn, tgt_conn, table: str, key: str, cols: dict[str, int | None]) -> dict:
    """Which detectors flag a difference, and how long each took (ms, both sides)."""
    out, ms = {}, {}
    checks = {
        "rowcount": lambda c: rowcount(c, table),
        "aggregate": lambda c: aggregate_sig(c, table),
        "rowhash_naive": lambda c: _hash_multiset(c, table, key, _naive_expr(cols)),
        "rowhash": lambda c: _hash_multiset(c, table, key, _robust_expr(cols)),
    }
    for name, fn in checks.items():
        t0 = time.perf_counter()
        out[name] = fn(src_conn) != fn(tgt_conn)
        ms[name] = round((time.perf_counter() - t0) * 1000, 1)
    out["ms"] = ms
    return out
