"""Benchmark (1), step 1: run the preprint's `Llama 3 Agentic ACE` pipeline with GPT-4o and record its full trajectory.

Fidelity: the code is read from the preprint pipeline's notebook (`Llama 3 Agentic ACE/Agentics stock selection.ipynb`, cell 0) and
executed as-is. Exactly three edits are made, each asserted so they fail loudly if his code changes:
  1. the LLM: `ollama/llama3` -> `openai/gpt-4o-2024-08-06` (temperature 0, as in his code);
  2. the Ollama `base_url` line is removed;
  3. the tool rate limit 5/min (meant for a local model) -> `calls_per_minute`.
His prompts, agents, tools, 3 ACE iterations, 12-month training window and zero-shot step are unchanged.

What is recorded per month (git-ignored, under data_local/benchmarks/b1/<YYYY-MM>/):
  trajectory.jsonl   every LLM call (messages + response + usage), every tool call (args + output), task boundaries
  systematic_strategy_MM_YYYY_gpt4o.txt   same format as the preprint pipeline's output files (rulebook + final zero-shot strategy)
  meta.json          model, windows, timings, token usage
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import LOCAL, REPO, data_file, read_key

NB = REPO / "Llama 3 Agentic ACE" / "Agentics stock selection.ipynb"
MODEL = "openai/gpt-4o-2024-08-06"   # training cutoff Oct 2023 -> Jul/Aug 2024 are out of sample


def paper_preprint_source(model: str = MODEL, calls_per_minute: int = 60) -> str:
    src = "".join(json.loads(NB.read_text())["cells"][0]["source"])
    edits = [
        ('model="ollama/llama3", # Explicitly requesting llama3, not llama3.1', f'model="{model}",'),
        ('base_url="http://localhost:11434", # Default local Ollama port', ""),
        ("rate_limiter = RateLimiter(calls_per_minute=5)", f"rate_limiter = RateLimiter(calls_per_minute={calls_per_minute})"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, f"The preprint notebook changed; cannot find: {old!r}"
        src = src.replace(old, new)
    main = 'if __name__ == "__main__":'
    assert src.count(main) == 1
    return src[: src.index(main)]           # definitions only; we drive the monthly loop below


def build_databases(workdir: Path) -> None:
    """Build firm_data.db / sentiment_data.db exactly as the preprint notebook's setup functions do, from the new data files."""
    workdir.mkdir(parents=True, exist_ok=True)
    if (workdir / "firm_data.db").exists() and (workdir / "sentiment_data.db").exists():
        return
    ns: dict = {}
    exec(paper_preprint_source(), ns)   # defines setup_firm_database / setup_sentiment_database (and agents; no LLM calls)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        ns["setup_firm_database"](str(data_file("green cleaned new.csv")))
        ns["setup_sentiment_database"](str(data_file("ravenpack_sp500_permno_final.csv")))
    finally:
        os.chdir(cwd)


class Recorder:
    """Subscribes to CrewAI's event bus and writes a JSONL trajectory."""

    def __init__(self, path: Path):
        self.path, self.lock, self.n = path, threading.Lock(), 0
        self.phase = "setup"
        self.f = open(path, "w")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def write(self, kind: str, **rec):
        with self.lock:
            if self.f.closed:          # CrewAI's bus is global: an earlier month's handlers still fire; ignore them
                return
            self.n += 1
            self.f.write(json.dumps({"step": self.n, "ts": time.time(), "phase": self.phase, "kind": kind, **rec},
                                    default=str) + "\n")
            self.f.flush()

    def attach(self):
        from crewai.events import (LLMCallCompletedEvent, LLMCallFailedEvent, TaskCompletedEvent, TaskStartedEvent,
                                   ToolUsageErrorEvent, ToolUsageFinishedEvent, crewai_event_bus)

        def role(ev):
            a = getattr(ev, "from_agent", None) or getattr(ev, "agent", None)
            return getattr(a, "role", None) or getattr(ev, "agent_role", None)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _llm(source, ev):
            u = ev.usage or {}
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
            self.write("llm_call", agent=role(ev), model=ev.model, messages=ev.messages, response=ev.response,
                       usage=u, finish_reason=ev.finish_reason)

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _llm_fail(source, ev):
            self.write("llm_error", agent=role(ev), error=str(getattr(ev, "error", "")))

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _tool(source, ev):
            self.write("tool_call", agent=ev.agent_role, tool=ev.tool_name, args=ev.tool_args, output=ev.output,
                       from_cache=ev.from_cache)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _tool_err(source, ev):
            self.write("tool_error", agent=ev.agent_role, tool=ev.tool_name, args=ev.tool_args, error=str(ev.error))

        @crewai_event_bus.on(TaskStartedEvent)
        def _task_start(source, ev):
            t = ev.task
            self.write("task_started", agent=getattr(getattr(t, "agent", None), "role", None),
                       description=getattr(t, "description", None))

        @crewai_event_bus.on(TaskCompletedEvent)
        def _task_done(source, ev):
            out = getattr(ev, "output", None)
            self.write("task_completed", agent=getattr(out, "agent", None), output=getattr(out, "raw", str(out)))

        self._handlers = (_llm, _llm_fail, _tool, _tool_err, _task_start, _task_done)   # keep references alive

    def close(self):
        self.f.close()


def windows(eval_date: str) -> tuple[str, str]:
    """The preprint pipeline's training window: 12 months ending the month before the evaluation month."""
    eval_dt = pd.Timestamp(eval_date)
    train_end_dt = eval_dt.replace(day=1) - pd.Timedelta(days=1)
    train_start_dt = train_end_dt - pd.DateOffset(months=12)
    return train_start_dt.strftime("%Y-%m-%d"), train_end_dt.strftime("%Y-%m-%d")


def run_month(eval_date: str, model: str = MODEL, iterations: int = 3, calls_per_minute: int = 60) -> Path:
    """Run training (ACE x iterations) + zero-shot evaluation for one month; returns the output directory."""
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ["OPENAI_API_KEY"] = read_key("openai")           # env var or key file; never printed or logged

    work = LOCAL / "b1" / "db"
    build_databases(work)
    out = LOCAL / "b1" / pd.Timestamp(eval_date).strftime("%Y-%m")
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out / "trajectory.jsonl")
    rec.attach()

    ns: dict = {}
    cwd = os.getcwd()
    t0 = time.time()
    try:
        os.chdir(work)                                   # the pipeline's tools open 'firm_data.db' / 'sentiment_data.db'
        exec(paper_preprint_source(model, calls_per_minute), ns)
        S = ns["PIPELINE_STATE"]
        S["current_eval_date"] = eval_date
        training_start, training_end = windows(eval_date)
        rec.phase = "training"
        S["allow_returns"] = True
        rulebook = ns["run_ace_loop"](training_start, training_end, iterations=iterations)
        rec.phase = "zero_shot_eval"
        S["allow_returns"] = False
        final = ns["generate_zero_shot_signals"](rulebook, eval_date)
    finally:
        os.chdir(cwd)
        rec.phase = "done"
        rec.close()
    fn = out / f"systematic_strategy_{pd.Timestamp(eval_date).strftime('%m_%Y')}_gpt4o.txt"
    fn.write_text(f"--- OPTIMIZED ACE RULEBOOK ---\n{rulebook}\n\n--- FINAL ZERO-SHOT STRATEGY ---\n{final}")
    meta = dict(eval_date=eval_date, training_start=training_start, training_end=training_end, model=model,
                iterations=iterations, seconds=round(time.time() - t0, 1), usage=rec.usage, steps=rec.n,
                source_notebook=str(NB.relative_to(REPO)), run_at=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
