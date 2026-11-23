"""Shared plumbing for benchmarks (1) and (2): paths, API keys, a read-only SQL tool, and a logged Claude tool loop.

Design choices
- The judge / explainer is Claude (claude-opus-5-5), a different model family from the GPT-4o system being judged,
  to avoid self-preference.
- Every request, tool call and tool result is appended to a JSONL transcript, so a human can audit the judge too.
- Tools are local and read-only (SQL over the same data the system under test saw). Nothing is executed remotely.
- Final answers are JSON constrained by a schema (structured outputs), so scores and claims can be checked by code.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                        # the Agentics repo root
PROJECT = REPO.parents[1]                 # optional folder two levels up, for key files / data kept outside the repo
LOCAL = REPO / "data_local" / "benchmarks"  # git-ignored: raw transcripts contain licensed data
RESULTS = HERE / "results"                # small, shareable outputs (scores, claims)

JUDGE_MODEL = "claude-opus-5-5"

# Folders searched for the licensed CSVs, in order. Set AGENTIC_DATA_DIR to override.
#   <repo>/Data              where the pipeline notebooks look ("../Data/green cleaned new.csv")
#   <repo root>              where `green cleaned.csv` goes
#   ../Data, <project>/data  data kept outside the repo
DATA_DIRS = [Path(p) for p in [os.environ.get("AGENTIC_DATA_DIR")] if p] + [
    REPO / "Data", REPO, REPO.parent / "Data", PROJECT / "data"]


def data_file(name: str) -> Path:
    """Locate a licensed data file (not in git) in the first folder of DATA_DIRS that has it."""
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name!r} not found. Looked in: " + ", ".join(str(d) for d in DATA_DIRS)
                            + ". Put it in one of these folders or set AGENTIC_DATA_DIR.")


_KEY_ENV = {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def read_key(provider: str) -> str:
    """API key from the environment variable (e.g. OPENAI_API_KEY), else from a <provider>_api_key.txt file
    outside the repo (<project>/ or the folder that contains the repo). Never logged, never committed."""
    env = _KEY_ENV[provider]
    if os.environ.get(env):
        return os.environ[env].strip()
    for p in [PROJECT / f"{provider}_api_key.txt", REPO.parent / f"{provider}_api_key.txt"]:
        if p.exists():
            return p.read_text().strip()
    raise RuntimeError(f"No {provider} API key: set {env}, or put the key in {REPO.parent / f'{provider}_api_key.txt'}")


# ---------------------------------------------------------------------------------------------------------------
# Read-only SQL tool
# ---------------------------------------------------------------------------------------------------------------
_WRITE = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export)\b", re.I)


@dataclass
class SQLTool:
    """A read-only SQL tool over a DuckDB or SQLite connection. Results are capped so they fit in context."""
    con: Any
    name: str
    description: str
    max_rows: int = 60
    max_chars: int = 8000

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "strict": True,
                "input_schema": {"type": "object", "additionalProperties": False, "required": ["query"],
                                 "properties": {"query": {"type": "string", "description": "One read-only SELECT (or WITH ... SELECT) statement."}}}}

    def __call__(self, query: str) -> tuple[str, bool]:
        q = query.strip().rstrip(";")
        if not re.match(r"^(select|with)\b", q, re.I) or _WRITE.search(q) or ";" in q:
            return "Error: only a single read-only SELECT / WITH query is allowed.", True
        try:
            if hasattr(self.con, "sql"):          # duckdb
                df = self.con.execute(q).df()
            else:                                  # sqlite3
                df = pd.read_sql_query(q, self.con)
        except Exception as e:                     # return the error to the model, like a real analyst would see it
            return f"Error: {type(e).__name__}: {e}", True
        n = len(df)
        txt = df.head(self.max_rows).to_string(index=False, max_colwidth=120)
        if len(txt) > self.max_chars:
            txt = txt[: self.max_chars] + "\n...[truncated]"
        more = f"\n...({n - self.max_rows} more rows not shown; aggregate in SQL instead)" if n > self.max_rows else ""
        return f"{n} rows\n{txt}{more}", False


# ---------------------------------------------------------------------------------------------------------------
# Logged Claude tool loop with a JSON-schema final answer
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class AgentRun:
    result: dict | None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    log_path: Path | None = None


def _dump(blocks) -> list[dict]:
    out = []
    for b in blocks:
        d = b.model_dump(mode="json", exclude_none=True) if hasattr(b, "model_dump") else dict(b)
        out.append(d)
    return out


def run_claude(system: str, user: str, tools: list[SQLTool], output_schema: dict, log_path: Path,
               model: str = JUDGE_MODEL, effort: str = "high", max_turns: int = 40,
               on_step: Callable[[str], None] | None = print) -> AgentRun:
    """Run Claude with local tools until it returns a schema-valid JSON answer. Everything is logged to `log_path`."""
    import anthropic

    client = anthropic.Anthropic(api_key=read_key("claude"))
    tool_map = {t.name: t for t in tools}
    messages: list[dict] = [{"role": "user", "content": user}]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write(json.dumps({"event": "start", "model": model, "effort": effort, "system": system, "user": user,
                          "tools": [t.schema() for t in tools], "output_schema": output_schema}) + "\n")
    run = AgentRun(result=None, log_path=log_path)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    try:
        for turn in range(max_turns):
            t0 = time.time()
            resp = client.beta.messages.create(
                model=model, max_tokens=16000, system=system, messages=messages,
                tools=[t.schema() for t in tools],
                thinking={"type": "adaptive", "display": "summarized"},   # keep a readable summary of the judge's reasoning
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": output_schema}},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # refusal -> server-side fallback model
            )
            for k in usage:
                usage[k] += getattr(resp.usage, k, 0) or 0
            run.models_used.append(resp.model)
            log.write(json.dumps({"event": "response", "turn": turn, "secs": round(time.time() - t0, 1), "model": resp.model,
                                  "stop_reason": resp.stop_reason, "content": _dump(resp.content),
                                  "usage": resp.usage.model_dump(mode="json")}) + "\n")
            run.stop_reason = resp.stop_reason
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                if resp.stop_reason == "max_tokens":
                    raise RuntimeError("hit max_tokens before finishing; raise max_tokens or lower effort")
                text = next(b.text for b in resp.content if b.type == "text")
                run.result = json.loads(text)
                break
            results = []
            for c in calls:
                tool = tool_map.get(c.name)
                q = c.input.get("query") if isinstance(c.input, dict) else None
                out, err = (tool(q) if tool and isinstance(q, str) else (f"Error: unknown tool or bad input {c.input!r}", True))
                rec = {"id": c.id, "tool": c.name, "query": q, "is_error": err, "output": out}
                run.tool_calls.append(rec)
                log.write(json.dumps({"event": "tool", "turn": turn, **rec}) + "\n")
                if on_step:
                    on_step(f"  [{turn:02d}] {c.name}: {(q or '')[:110]!s}{' -> ERROR' if err else ''}")
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": out, **({"is_error": True} if err else {})})
            messages.append({"role": "user", "content": results})
        else:
            raise RuntimeError(f"no final answer after {max_turns} turns")
    finally:
        run.usage = usage
        log.write(json.dumps({"event": "end", "usage": usage, "models_used": run.models_used, "result": run.result}) + "\n")
        log.close()
    return run


def cost_usd(usage: dict, model: str = JUDGE_MODEL) -> float:
    """Rough cost at list prices (Opus 5.5: $4 in, $20 out, $0.20 cache read, $5 cache write (1.25x) per MTok)."""
    price = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}.get(model, (4.0, 20.0, 0.20, 5.0))
    return (usage.get("input_tokens", 0) * price[0] + usage.get("output_tokens", 0) * price[1]
            + usage.get("cache_read_input_tokens", 0) * price[2]
            + usage.get("cache_creation_input_tokens", 0) * price[3]) / 1e6
