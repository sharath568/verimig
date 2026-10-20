"""SQL Server execution helpers: batch splitting, isolated scratch databases,
and execution that captures result sets, errors and informational messages."""
from __future__ import annotations

import contextlib
import datetime as _dt
import decimal
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_GO = re.compile(r"^\s*go\s*;?\s*$", re.IGNORECASE | re.MULTILINE)
_FENCE = re.compile(r"```(?:sql|tsql|t-sql)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

SESSION_SETTINGS = (
    "SET ANSI_NULLS ON; SET QUOTED_IDENTIFIER ON; SET CONCAT_NULL_YIELDS_NULL ON; "
    "SET ANSI_WARNINGS ON; SET ANSI_PADDING ON; SET IMPLICIT_TRANSACTIONS OFF; SET NOCOUNT ON;"
)


def split_batches(sql: str) -> list[str]:
    """Split a script on GO separators (client-side, as SSMS/sqlcmd do)."""
    parts = [p.strip() for p in _GO.split(sql or "")]
    return [p for p in parts if p]


def extract_sql(text: str) -> str:
    """Pull SQL out of an LLM reply (prefers fenced blocks)."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return "\nGO\n".join(b.strip() for b in blocks)
    return (text or "").strip()


def normalize_value(v: Any) -> Any:
    """Canonical, LOSSLESS text form of a value for comparison.

    Numbers keep their exact value: trailing zeros are dropped (100.0000 == 100) but no rounding is
    applied, so 1.0000001 and 1.0000002 stay different. A case may declare an explicit numeric
    tolerance (field `tolerance`) when a difference is acceptable; nothing is tolerated by default.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, decimal.Decimal)):
        d = decimal.Decimal(v)
        if d == 0:
            return "0"
        return format(d.normalize(), "f")
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    return str(v)


@dataclass
class ExecOutcome:
    resultsets: list[list[tuple]] = field(default_factory=list)
    columns: list[list[str]] = field(default_factory=list)
    error: tuple[int, str] | None = None
    messages: list[str] = field(default_factory=list)
    tran: list | None = None  # (@@TRANCOUNT, XACT_STATE()) observed right after the call
    coltypes: list = field(default_factory=list)  # driver type code per column, per result set

    def to_dict(self) -> dict:
        return {
            "resultsets": [[list(r) for r in rs] for rs in self.resultsets],
            "columns": self.columns,
            "error": list(self.error) if self.error else None,
            "messages": self.messages,
            "tran": self.tran,
        }


def _clean_error(exc: Exception) -> tuple[int, str]:
    num, msg = -1, str(exc)
    args = getattr(exc, "args", ())
    if len(args) >= 2:
        num = args[0] if isinstance(args[0], int) else -1
        raw = args[1]
        msg = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    # pymssql appends DB-Lib boilerplate; keep the server text only.
    msg = msg.split("DB-Lib error message")[0].strip()
    return num, msg


class SqlServer:
    """Thin wrapper over pymssql with per-evaluation scratch databases."""

    def __init__(self, host: str, port: int, user: str, password: str, timeout: int = 30):
        self.host, self.port, self.user, self.password, self.timeout = host, port, user, password, timeout

    def connect(self, database: str = "master"):
        import pymssql  # imported lazily so offline tests run without it

        conn = pymssql.connect(
            server=self.host, port=self.port, user=self.user, password=self.password,
            database=database, autocommit=True, login_timeout=15, timeout=self.timeout,
        )
        return conn

    @contextlib.contextmanager
    def scratch_db(self, prefix: str = "vm"):
        name = f"{prefix}_{uuid.uuid4().hex[:12]}"
        admin = self.connect("master")
        try:
            admin.cursor().execute(f"CREATE DATABASE [{name}]")
            conn = self.connect(name)
            try:
                conn.cursor().execute(SESSION_SETTINGS)
                yield conn
            finally:
                conn.close()
        finally:
            with contextlib.suppress(Exception):
                admin.cursor().execute(
                    f"ALTER DATABASE [{name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [{name}]"
                )
            admin.close()

    @staticmethod
    def run_script(conn, script: str) -> tuple[bool, str | None]:
        """Execute every GO-separated batch; stop at the first error."""
        cur = conn.cursor()
        for batch in split_batches(script):
            try:
                cur.execute(batch)
                while cur.nextset():
                    pass
            except Exception as exc:  # noqa: BLE001
                return False, _clean_error(exc)[1]
        return True, None

    @staticmethod
    def alive(conn) -> bool:
        """False once the driver has dropped the connection (e.g. after a statement timeout)."""
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def signatures(conn) -> dict:
        """Parameter signature of every user procedure: [(name, type, length, precision, scale, is_output)]."""
        cur = conn.cursor()
        cur.execute(
            "SELECT o.name, p.name, TYPE_NAME(p.user_type_id), p.max_length, p.precision, p.scale, p.is_output "
            "FROM sys.objects o LEFT JOIN sys.parameters p ON p.object_id = o.object_id "
            "WHERE o.type = 'P' AND o.is_ms_shipped = 0 ORDER BY o.name, p.parameter_id")
        sig: dict[str, list] = {}
        for proc, *rest in cur.fetchall():
            sig.setdefault(proc, [])
            if rest[0] is not None:
                sig[proc].append(tuple(str(x) for x in rest))
        return sig

    @staticmethod
    def snapshot(conn) -> dict:
        """Final contents of every user table, as sorted lists of normalised rows.

        rowversion/timestamp columns are excluded: their values are database counters, not data.
        """
        cur = conn.cursor()
        cur.execute(
            "SELECT t.name, c.name FROM sys.tables t JOIN sys.columns c ON c.object_id = t.object_id "
            "WHERE t.is_ms_shipped = 0 AND c.system_type_id <> 189 ORDER BY t.name, c.column_id")
        cols: dict[str, list[str]] = {}
        for t, c in cur.fetchall():
            cols.setdefault(t, []).append(c)
        state = {}
        for t, cs in cols.items():
            cur.execute("SELECT " + ", ".join(f"[{c}]" for c in cs) + f" FROM [{t}]")
            state[t] = sorted((tuple(normalize_value(v) for v in row) for row in cur.fetchall()), key=repr)
        return state

    @staticmethod
    def execute_capture(conn, sql: str) -> ExecOutcome:
        out = ExecOutcome()

        def _handler(msgstate, severity, srvname, procname, line, msgtext):
            text = msgtext.decode("utf-8", "replace") if isinstance(msgtext, bytes) else str(msgtext)
            if severity <= 10:
                out.messages.append(text.strip())

        with contextlib.suppress(Exception):
            conn._conn.set_msghandler(_handler)  # pymssql internal API; optional
        cur = conn.cursor()
        try:
            cur.execute(sql)
            while True:
                if cur.description:
                    out.columns.append([d[0] for d in cur.description])
                    out.coltypes.append([d[1] for d in cur.description])
                    out.resultsets.append([tuple(normalize_value(v) for v in row) for row in cur.fetchall()])
                if not cur.nextset():
                    break
        except Exception as exc:  # noqa: BLE001
            out.error = _clean_error(exc)
        finally:
            with contextlib.suppress(Exception):
                conn._conn.set_msghandler(None)
        return out
