"""Gate G4: data reconciliation between a source and a target table.

Three detectors of increasing strength, all expressed in plain T-SQL so they run on SQL Server
(and, with minor edits, on ASE):

  rowcount      COUNT(*) only (the most common check in practice)
  aggregate     COUNT + CHECKSUM_AGG(BINARY_CHECKSUM(*)) over the table
  rowhash       per-key SHA2-256 over a canonical, NULL-safe serialisation of every column
"""
from __future__ import annotations


def rowcount(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT_BIG(*) FROM {table}")
    return cur.fetchone()[0]


def aggregate_sig(conn, table: str) -> tuple:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT_BIG(*), CHECKSUM_AGG(BINARY_CHECKSUM(*)) FROM {table}")
    return tuple(cur.fetchone())


def _ser(col: str, style: int | None) -> str:
    conv = f"CONVERT(nvarchar(4000), {col}, {style})" if style is not None else f"CONVERT(nvarchar(4000), {col})"
    return f"ISNULL({conv}, N'<NULL>')"


def rowhash_map(conn, table: str, key: str, cols: dict[str, int | None]) -> dict:
    """cols maps column -> CONVERT style (126 for datetime types, 2 for money, None otherwise)."""
    parts = " + N'|' + ".join(_ser(c, st) for c, st in cols.items())
    cur = conn.cursor()
    cur.execute(f"SELECT {key}, COUNT(*) OVER (PARTITION BY {key}), "
                f"CONVERT(varchar(64), HASHBYTES('SHA2_256', {parts}), 2) FROM {table}")
    out: dict = {}
    for k, dup, h in cur.fetchall():
        out[k] = (dup, h) if k not in out else (dup, out[k][1] + "+" + h)
    return out


def detect(src_conn, tgt_conn, table: str, key: str, cols: dict[str, int | None]) -> dict:
    """Return which detectors flag a difference between source and target."""
    rc = rowcount(src_conn, table) != rowcount(tgt_conn, table)
    ag = aggregate_sig(src_conn, table) != aggregate_sig(tgt_conn, table)
    rh = rowhash_map(src_conn, table, key, cols) != rowhash_map(tgt_conn, table, key, cols)
    return {"rowcount": rc, "aggregate": ag, "rowhash": rh}
