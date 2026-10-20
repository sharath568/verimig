"""Converters: each returns a ConversionResult for one corpus case.

  rules        deterministic regex baseline (intentionally simple, documented in the paper)
  imported     reads results/imported/<tool>/<case_id>.sql (e.g. SSMA output you export)
  llm_plain    local LLM, generic translation prompt
  llm_trap     local LLM, prompt augmented with the dialect-trap checklist
  agentic      llm_trap + repair loop driven only by G1 (compile) and G2 (lint) feedback
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..sqlutil import extract_sql
from ..taxonomy import trap_checklist


@dataclass
class ConversionResult:
    sql: str
    attempts: int = 1
    latency_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    gen_tokens_per_s: float | None = None
    trace: list = field(default_factory=list)


# --------------------------------------------------------------------------- rules baseline
_RULES: list[tuple[str, str]] = [
    (r"\bstr_replace\s*\(", "REPLACE("),
    (r"@@sqlstatus\s*=\s*0", "@@FETCH_STATUS = 0"),
    (r"@@sqlstatus\s*(!=|<>)\s*0", "@@FETCH_STATUS <> 0"),
    (r"\bdeallocate\s+cursor\s+", "DEALLOCATE "),
    (r"\b(where|and|or)(\s+[\w.@]+\s*)=\s*null\b", r"\1\2IS NULL"),
    (r"\b(where|and|or)(\s+[\w.@]+\s*)(!=|<>)\s*null\b", r"\1\2IS NOT NULL"),
    (r"\bfrom\s+(\w+)(\s+\w+)?\s+holdlock\b", r"FROM \1\2 WITH (HOLDLOCK)"),
    (r"\bfrom\s+(\w+)(\s+\w+)?\s+readpast\b", r"FROM \1\2 WITH (READPAST)"),
    (r"\s+noholdlock\b", ""),
    (r"\(\s*index\s+(\w+)\s*\)", r"WITH (INDEX(\1))"),
    (r"\bunsigned\s+int\b", "bigint"),
    (r"\bunsigned\s+bigint\b", "decimal(20,0)"),
    (r"\bunsigned\s+smallint\b", "int"),
    (r"\bbigdatetime\b", "datetime2(6)"),
    (r"\bbigtime\b", "time(6)"),
    (r"\bunivarchar\b", "nvarchar"),
    (r"\bunichar\b", "nchar"),
    (r"\bunitext\b", "nvarchar(max)"),
    (r"\s+lock\s+(datarows|datapages|allpages)\b", ""),
    (r"\s+with\s+(max_rows_per_page|exp_row_size|reservepagegap)\s*=\s*\d+", ""),
    (r"@@tranchained", "((@@OPTIONS & 2) / 2)"),
    (r'\braiserror\s+\d+\s+"([^"]*)"', r"RAISERROR('\1', 16, 1)"),
]


def _dq_to_sq(sql: str) -> str:
    # Convert "literal text" (contains space/punctuation) to 'literal text'.
    return re.sub(r'"([^"\n]*[\s,:!%.][^"\n]*)"', lambda m: "'" + m.group(1).replace("'", "''") + "'", sql)


def convert_rules(source: str) -> ConversionResult:
    t0 = time.perf_counter()
    sql = source
    for pat, rep in _RULES:
        sql = re.sub(pat, rep, sql, flags=re.IGNORECASE)
    sql = _dq_to_sq(sql)
    return ConversionResult(sql=sql, latency_s=time.perf_counter() - t0)


# --------------------------------------------------------------------------- imported (SSMA etc.)
def convert_imported(case_id: str, tool_dir: Path) -> ConversionResult | None:
    p = tool_dir / f"{case_id}.sql"
    if not p.exists():
        return None
    return ConversionResult(sql=p.read_text())


# --------------------------------------------------------------------------- LLM (Ollama)
SYSTEM_PLAIN = (
    "You are an expert database migration engineer. Translate SAP ASE (Sybase) Transact-SQL into "
    "Microsoft SQL Server 2022 Transact-SQL. Preserve the runtime behaviour of the original exactly "
    "(same result sets, same NULL handling, same errors raised). Keep object names unchanged. "
    "Return only the translated code in a single ```sql fenced block, using GO between batches if needed. "
    "Do not add explanations, test data or extra objects."
)

SYSTEM_TRAP = SYSTEM_PLAIN + (
    "\n\nCheck the code against this list of known ASE -> SQL Server dialect traps before answering:\n"
    + trap_checklist()
)


class Ollama:
    def __init__(self, url: str, temperature: float = 0.2, num_ctx: int = 8192, timeout: int = 600):
        self.url, self.temperature, self.num_ctx, self.timeout = url.rstrip("/"), temperature, num_ctx, timeout

    def chat(self, model: str, messages: list[dict], seed: int) -> dict:
        r = requests.post(
            f"{self.url}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": self.temperature, "seed": seed, "num_ctx": self.num_ctx}},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


class Anthropic:
    dropped_temperature = False
    """Hosted reference model via the Anthropic Messages API (key from ANTHROPIC_API_KEY).

    Same interface as Ollama.chat. The API has no seed parameter, so repeated runs may differ more
    than local runs; temperature matches the local setting.
    """

    def __init__(self, temperature: float = 0.2, timeout: int = 600, max_tokens: int = 2048):
        import os
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (add it to .env)")
        self.temperature, self.timeout, self.max_tokens = temperature, timeout, max_tokens

    def chat(self, model: str, messages: list[dict], seed: int) -> dict:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        name = model.split(":", 1)[1] if ":" in model else model
        t0 = time.perf_counter()
        body = {"model": name, "max_tokens": self.max_tokens, "messages": conv}
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        r = None
        for attempt in range(6):
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=self.timeout, headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json=body)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            if r.status_code == 400 and "temperature" in body and "temperature" in r.text.lower():
                # some models fix their own sampling; fall back to the default and remember it
                body.pop("temperature")
                self.temperature = None
                Anthropic.dropped_temperature = True
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Anthropic API {r.status_code}: {r.text[:300]}")
            break
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        dur = time.perf_counter() - t0
        return {"message": {"content": text}, "prompt_eval_count": usage.get("input_tokens", 0),
                "eval_count": usage.get("output_tokens", 0), "eval_duration": int(dur * 1e9),
                "temperature_used": self.temperature}


def is_hosted(model: str) -> bool:
    return model.startswith("anthropic:")


def _user_msg(source: str) -> str:
    return f"Translate this SAP ASE code to SQL Server:\n```sql\n{source.strip()}\n```"


def _accumulate(res: ConversionResult, reply: dict):
    res.prompt_tokens += reply.get("prompt_eval_count", 0) or 0
    res.output_tokens += reply.get("eval_count", 0) or 0
    ev, dur = reply.get("eval_count"), reply.get("eval_duration")
    if ev and dur:
        res.gen_tokens_per_s = ev / (dur / 1e9)


def convert_llm(client: Ollama, model: str, source: str, seed: int, trap_aware: bool) -> ConversionResult:
    msgs = [{"role": "system", "content": SYSTEM_TRAP if trap_aware else SYSTEM_PLAIN},
            {"role": "user", "content": _user_msg(source)}]
    t0 = time.perf_counter()
    reply = client.chat(model, msgs, seed)
    res = ConversionResult(sql=extract_sql(reply["message"]["content"]))
    res.latency_s = time.perf_counter() - t0
    _accumulate(res, reply)
    res.trace.append({"attempt": 1, "sql": res.sql})
    return res


def convert_agentic(client: Ollama, model: str, source: str, seed: int,
                    compile_fn, lint_fn, max_iters: int = 3) -> ConversionResult:
    """Translate, then repair using ONLY compiler errors (G1) and residual-construct lint (G2)."""
    msgs = [{"role": "system", "content": SYSTEM_TRAP}, {"role": "user", "content": _user_msg(source)}]
    res = ConversionResult(sql="", attempts=0)
    t0 = time.perf_counter()
    for attempt in range(1, max_iters + 1):
        reply = client.chat(model, msgs, seed + attempt - 1)
        _accumulate(res, reply)
        content = reply["message"]["content"]
        sql = extract_sql(content)
        res.sql, res.attempts = sql, attempt
        ok_compile, err = compile_fn(sql)
        ok_lint, findings = lint_fn(sql)
        res.trace.append({"attempt": attempt, "sql": sql, "g1": ok_compile, "g1_error": err,
                          "g2": ok_lint, "lint": [f["rule"] for f in findings if f["severity"] == "error"]})
        if ok_compile and ok_lint:
            break
        feedback = []
        if not ok_compile:
            feedback.append(f"SQL Server rejected the code with this error:\n{err}")
        if not ok_lint:
            feedback.append("These ASE constructs remain and must be converted:\n" + "\n".join(
                f"- {f['match']!r}: {f['hint']}" for f in findings if f["severity"] == "error"))
        msgs += [{"role": "assistant", "content": content},
                 {"role": "user", "content": "\n\n".join(feedback)
                  + "\n\nReturn the corrected full translation in a single ```sql block."}]
    res.latency_s = time.perf_counter() - t0
    return res
