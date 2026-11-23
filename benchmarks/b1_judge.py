"""Benchmark (1), step 2: deterministic audit + LLM judge of the GPT-4o ACE trajectory ("chain of thought").

Stage-1 discipline (Capponi's email): the judge scores the PROCESS before outcomes are revealed. It never sees realized
returns of the decision month: its SQL tool masks ret_fwd_1 for every fundamentals row dated >= the evaluation date and
every news row after the training window. Realized outcomes are computed separately (`outcome()`), so we can later test
whether process scores predict performance without the score having seen it.

Pieces
- `load_trajectory(month)`   the JSONL recorded by b1_ace_pipeline
- `render(traj)`             readable transcript with step ids [S012] (prompts once per agent, then every call/result)
- `audit(traj, meta)`        deterministic facts: tool counts/errors, look-ahead, use of ret_fwd_1, prompt-example
                             copying, and whether the final rule is executable (counts of BUY/SELL names)
- `judge(month)`             Claude scores the rubric RUBRIC (0/1/2 per item, with evidence step ids)
- `outcome(month)`           realized next-month returns of the final BUY / SELL lists (kept out of the judge)
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import LOCAL, RESULTS, SQLTool, run_claude
from .b1_ace_pipeline import build_databases, paper_preprint_source

B1 = LOCAL / "b1"

# ---------------------------------------------------------------------------------------------------------------
# Rubric (0 = missing or incorrect, 1 = partially supported, 2 = fully supported)
# ---------------------------------------------------------------------------------------------------------------
RUBRIC = [
    ("P1", "Data exploration",
     "2: inspected schemas and the distributions/ranges relevant to the thresholds it later chose, within the training window. "
     "1: some exploration but not of the variables/ranges that matter. 0: none, or exploration failed and was not recovered."),
    ("P2", "Hypotheses tested against outcomes",
     "2: candidate rules were tested against realized forward returns (ret_fwd_1) in the training window, e.g. average "
     "return or hit rate of BUY vs SELL names, and the results informed the choice. 1: only counted how many firms pass, "
     "or tested returns but ignored the result. 0: no testing against outcomes."),
    ("P3", "Rule traceable to evidence",
     "2: each threshold in the final rule can be traced to a specific tool result (cite steps). 1: some thresholds traced, "
     "others arbitrary. 0: thresholds are arbitrary or copied from the prompt's examples."),
    ("P4", "Tool use",
     "2: tool calls were valid, relevant, non-redundant, errors were diagnosed and recovered, and the tools changed what "
     "the agent concluded. 1: tools were used but errors went unrecovered or results were not used. 0: tools not used or misused."),
    ("P5", "Fundamentals-sentiment integration",
     "2: the combined rule's structure (AND vs OR, which domain dominates) is justified by evidence about how the two "
     "signals interact. 1: combined, but the combination is asserted, not argued from data. 0: no real integration."),
    ("P6", "Rule well-formed and executable",
     "2: unambiguous precedence, uses only available variables, BUY and SELL disjoint and both non-empty on the decision "
     "date. 1: executable with minor ambiguity or a near-empty side. 0: not executable, contradictory, or ill-defined."),
    ("P7", "Temporal discipline (no look-ahead)",
     "2: training used only data inside the training window, and evaluation used only data up to the decision date, "
     "without forward returns. 1: minor or ambiguous excursions (e.g. undated whole-table queries). 0: used future data."),
    ("P8", "Fidelity of the final (zero-shot) step",
     "2: the final step applied the curated rulebook as written and output explicit thresholds. 1: minor unexplained "
     "changes. 0: rewrote the rule, refused, or did not produce explicit thresholds."),
    ("P9", "Rationale consistent with evidence",
     "2: the economic rationale matches the rule and the data the agent saw; no unsupported statistics. 1: generic "
     "rationale, or some claims not supported by tool output. 0: rationale contradicts the rule or invents facts."),
]

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["items", "tools_used_summary", "tools_helpful_score", "tools_helpful_explanation", "overall_score",
                 "overall_summary", "strengths", "weaknesses", "red_flags"],
    "properties": {
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "score", "justification", "evidence_steps", "verification"],
            "properties": {
                "id": {"type": "string", "enum": [r[0] for r in RUBRIC]},
                "score": {"type": "integer", "enum": [0, 1, 2]},
                "justification": {"type": "string"},
                "evidence_steps": {"type": "array", "items": {"type": "string"}, "description": "step ids like S012"},
                "verification": {"type": "string", "description": "what you re-checked with SQL and what you found, or 'none'"},
            }}},
        "tools_used_summary": {"type": "string", "description": "which tools were used, how, and to what end"},
        "tools_helpful_score": {"type": "integer", "enum": [0, 1, 2]},
        "tools_helpful_explanation": {"type": "string"},
        "overall_score": {"type": "integer", "enum": [0, 1, 2]},
        "overall_summary": {"type": "string", "description": "How well did the system arrive at its decision rule?"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
    },
}

JUDGE_SYSTEM = """You are an expert reviewer of quantitative research processes, grading how an LLM agent system arrived
at a stock-screening rule. You grade the PROCESS, not the outcome: you are not told how the rule performed, and you
must not try to find out (forward returns at or after the decision date are masked in your database).

Grade strictly against the rubric anchors. For every item:
- cite the transcript step ids (e.g. S012) that support your score;
- re-check the agent's factual claims with your SQL tool where you can (e.g. re-run a condition count, check whether a
  threshold matches any tool output, check whether a query's dates were inside the allowed window);
- prefer the deterministic audit facts over your own impression when they conflict, and say so.
Be concrete and terse. A score of 2 needs positive evidence; absence of evidence is not evidence."""


# ---------------------------------------------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------------------------------------------
def load_trajectory(month: str) -> tuple[list[dict], dict]:
    d = B1 / month
    traj = [json.loads(l) for l in open(d / "trajectory.jsonl")]
    return traj, json.loads((d / "meta.json").read_text())


def _resp_text(resp) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        calls = []
        for c in resp:
            f = c.get("function", {}) if isinstance(c, dict) else {}
            calls.append(f"{f.get('name')}({f.get('arguments')})")
        return "CALLS TOOLS: " + "; ".join(calls)
    return str(resp)


def render(traj: list[dict]) -> str:
    out, seen_prompt = [], set()
    for r in traj:
        sid, ph, ag = f"S{r['step']:03d}", r["phase"], r.get("agent") or "?"
        if r["kind"] == "llm_call":
            if ag not in seen_prompt and isinstance(r.get("messages"), list):
                seen_prompt.add(ag)
                for m in r["messages"][:2]:
                    out.append(f"----- PROMPT ({m.get('role')}) for agent '{ag}' (shown once) -----\n{m.get('content')}\n")
            out.append(f"[{sid}] ({ph}) {ag} -> {_resp_text(r.get('response'))}")
        elif r["kind"] == "tool_call":
            out.append(f"[{sid}] ({ph}) TOOL {r.get('tool')}({json.dumps(r.get('args'))})\n      RESULT: {r.get('output')}")
        elif r["kind"] == "tool_error":
            out.append(f"[{sid}] ({ph}) TOOL ERROR {r.get('tool')}({json.dumps(r.get('args'))}): {r.get('error')}")
        elif r["kind"] == "task_started":
            out.append(f"\n===== [{sid}] ({ph}) TASK START: {ag} =====\n{r.get('description')}")
        elif r["kind"] == "task_completed":
            out.append(f"===== [{sid}] ({ph}) TASK OUTPUT ({ag}) recorded =====")
        elif r["kind"] == "llm_error":
            out.append(f"[{sid}] ({ph}) LLM ERROR {ag}: {r.get('error')}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Deterministic audit
# ---------------------------------------------------------------------------------------------------------------
_COND = re.compile(r"\b(bm|mve|mom12m|avg_sentiment|news_count)\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def prompt_example_conditions() -> set[tuple]:
    """Every `var op number` that appears in the examples of the preprint pipeline's prompts."""
    src = paper_preprint_source()
    return {(v, o, float(n)) for v, o, n in _COND.findall(src)}


def extract_final_rule(text: str) -> dict:
    """BUY/SELL lines of section 3 (UNIFIED BOOLEAN INTEGRATION RULE) of the final zero-shot output."""
    sec = text.split("--- FINAL ZERO-SHOT STRATEGY ---")[-1]
    m = re.search(r"UNIFIED BOOLEAN INTEGRATION RULE.*?(?=\n\s*4\.|\Z)", sec, re.S | re.I)
    block = m.group(0) if m else sec
    rule = {}
    for side in ("BUY", "SELL"):
        mm = re.search(rf"{side}\s+if\s*:?\s*`?(.+?)`?\s*$", block, re.I | re.M)
        rule[side] = mm.group(1).strip().strip("`") if mm else None
    return rule


def cross_section(eval_date: str) -> pd.DataFrame:
    """Fundamentals at the decision date + trailing-30-day sentiment exactly as the pipeline's test_sentiment_condition."""
    db = B1 / "db"
    build_databases(db)                  # no-op if present; otherwise built from the CSVs (no LLM)
    f = pd.read_sql_query("SELECT ticker, mve, bm, mom12m, ret_fwd_1 FROM firms WHERE datadate = ?",
                          sqlite3.connect(db / "firm_data.db"), params=[eval_date])
    end = pd.Timestamp(eval_date)
    s = pd.read_sql_query(
        "SELECT ticker, AVG(event_sentiment_score) AS avg_sentiment, COUNT(clean_headline) AS news_count FROM ravenpack "
        "WHERE date BETWEEN ? AND ? GROUP BY ticker", sqlite3.connect(db / "sentiment_data.db"),
        params=[(end - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")])
    x = f.merge(s, on="ticker", how="left")
    x["news_count"] = x.news_count.fillna(0)
    return x


def eval_rule(expr: str, df: pd.DataFrame) -> pd.Series:
    py = re.sub(r"\bAND\b", " and ", expr, flags=re.I)
    py = re.sub(r"\bOR\b", " or ", py, flags=re.I)
    py = re.sub(r"\bNOT\b", " not ", py, flags=re.I)
    py = re.sub(r"(?<![<>=!])=(?!=)", "==", py)
    return df.eval(py, engine="python").fillna(False).astype(bool)


def audit(traj: list[dict], meta: dict) -> dict:
    tr_end, ev = meta["training_end"], meta["eval_date"]
    tools = [r for r in traj if r["kind"] == "tool_call"]
    llm = [r for r in traj if r["kind"] == "llm_call"]
    err = [t for t in tools if isinstance(t.get("output"), str) and ("Error" in t["output"][:40] or "SECURITY BLOCK" in t["output"])]
    look = []
    for t in tools:
        s = json.dumps(t.get("args"))
        limit = tr_end if t["phase"] == "training" else ev
        issues = []
        late = [d for d in _DATE.findall(s) if d > limit]
        if late:
            issues.append(f"query asks for dates after {limit}: {sorted(set(late))}")
        if t["tool"].startswith("query_") and not _DATE.findall(s):      # raw SQL with no date bound sees every date
            issues.append("raw SQL with no date filter (whole table, including future rows)"
                          + (" and it reads ret_fwd_1" if "ret_fwd_1" in s else ""))
        seen = [d for d in _DATE.findall(str(t.get("output") or "")) if d > limit]
        if seen:                                                          # the strongest evidence: future data was returned
            issues.append(f"tool OUTPUT contains rows dated after {limit} (latest {max(seen)})")
        if issues:
            look.append({"step": f"S{t['step']:03d}", "phase": t["phase"], "tool": t["tool"], "issue": "; ".join(issues)})
    fwd_tests = [f"S{t['step']:03d}" for t in tools if t["phase"] == "training" and "ret_fwd_1" in json.dumps(t.get("args"))]

    text = (B1 / pd.Timestamp(ev).strftime("%Y-%m") / f"systematic_strategy_{pd.Timestamp(ev).strftime('%m_%Y')}_gpt4o.txt").read_text()
    rule = extract_final_rule(text)
    ex = prompt_example_conditions()
    conds = {s: {(v, o, float(n)) for v, o, n in _COND.findall(rule[s] or "")} for s in ("BUY", "SELL")}
    copied = sorted({f"{v} {o} {n:g}" for s in conds for (v, o, n) in conds[s] if (v, o, n) in ex})
    allc = set().union(*conds.values())

    execu = {"BUY": None, "SELL": None, "overlap": None, "error": None}
    try:
        X = cross_section(ev)
        b, s_ = eval_rule(rule["BUY"], X), eval_rule(rule["SELL"], X)
        execu = {"universe": len(X), "BUY": int(b.sum()), "SELL": int(s_.sum()), "overlap": int((b & s_).sum()), "error": None}
    except Exception as e:                                         # an unparseable rule is itself a finding
        execu["error"] = f"{type(e).__name__}: {e}"
    by_agent = pd.Series([r.get("agent") for r in llm]).value_counts().to_dict()
    return {
        "eval_date": ev, "training_window": f"{meta['training_start']} .. {tr_end}", "model": meta["model"],
        "llm_calls_by_agent": by_agent,
        "tool_calls_by_tool": pd.Series([t["tool"] for t in tools]).value_counts().to_dict(),
        "tool_errors": [{"step": f"S{t['step']:03d}", "tool": t["tool"], "output": t["output"][:160]} for t in err],
        "possible_look_ahead": look,
        "training_queries_using_ret_fwd_1": fwd_tests,
        "final_rule": rule,
        "final_rule_conditions_copied_from_prompt_examples": copied,
        "share_of_final_conditions_copied": (len([c for c in allc if c in ex]) / len(allc)) if allc else None,
        "final_rule_on_decision_date": execu,
        "tokens": meta.get("usage"),
    }


def outcome(month: str) -> dict:
    """Realized next-month (excess) return of the final BUY / SELL names. NOT shown to the judge."""
    traj, meta = load_trajectory(month)
    a = audit(traj, meta)
    X = cross_section(meta["eval_date"])
    r = a["final_rule"]
    try:
        b, s = eval_rule(r["BUY"], X), eval_rule(r["SELL"], X)
    except Exception as e:
        return {"error": str(e)}
    u = X.ret_fwd_1.mean()
    return {"eval_date": meta["eval_date"], "return_month": (pd.Timestamp(meta["eval_date"]) + pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
            "n_buy": int(b.sum()), "n_sell": int(s.sum()), "buy_mean": X.ret_fwd_1[b].mean(), "sell_mean": X.ret_fwd_1[s].mean(),
            "universe_mean": u, "buy_minus_sell": X.ret_fwd_1[b].mean() - X.ret_fwd_1[s].mean(),
            "buy_hit_rate_vs_universe": float((X.ret_fwd_1[b] > u).mean()) if b.any() else np.nan}


# ---------------------------------------------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------------------------------------------
def judge_tools(meta: dict) -> list[SQLTool]:
    """The same two databases the agents used, with outcomes at/after the decision date masked."""
    db, ev, tr_end = B1 / "db", meta["eval_date"], meta["training_end"]
    build_databases(db)
    con = duckdb.connect()
    f = pd.read_sql_query("SELECT * FROM firms", sqlite3.connect(db / "firm_data.db"))
    f.loc[f.datadate >= ev, "ret_fwd_1"] = np.nan
    s = pd.read_sql_query("SELECT * FROM ravenpack", sqlite3.connect(db / "sentiment_data.db"))
    s.loc[s.date > tr_end, "ret_fwd_1"] = np.nan
    con.register("_f", f); con.execute("CREATE TABLE firms AS SELECT * FROM _f")
    con.register("_s", s); con.execute("CREATE TABLE ravenpack AS SELECT * FROM _s")
    return [SQLTool(con, "query_data",
                    "Read-only DuckDB SQL over two tables, the same data the agents used: "
                    "firms(PERMNO, ticker, comnam, datadate 'YYYY-MM-DD', ret_fwd_1, mve_m, mve, mom12m, bm) and "
                    "ravenpack(date 'YYYY-MM-DD', ticker, event_sentiment_score, clean_headline, category, ..., ret_fwd_1). "
                    f"ret_fwd_1 is masked (NULL) for firms rows dated >= {ev} and news after {tr_end}.")]


def judge_prompt(month: str) -> tuple[str, dict]:
    traj, meta = load_trajectory(month)
    a = audit(traj, meta)
    text = (B1 / month / f"systematic_strategy_{pd.Timestamp(meta['eval_date']).strftime('%m_%Y')}_gpt4o.txt").read_text()
    rubric = "\n".join(f"{i} {name}: {anchors}" for i, name, anchors in RUBRIC)
    user = f"""Grade the decision process of the agent system below for decision date {meta['eval_date']}.

## The system
The preprint's "Llama 3 Agentic ACE" pipeline, run here with {meta['model']} (temperature 0). Per month: 3 ACE iterations on a
training window ({meta['training_start']} .. {meta['training_end']}, forward returns visible), each = Fundamentals
generator -> Sentiment generator -> Reflector -> Curator (rulebook). Then a zero-shot Evaluator applies the final rulebook
on the decision date with forward returns blocked. The prompts each agent received are in the transcript (shown once per agent).

## Data the agents had
- firm_data.db / firms: S&P 500 monthly panel. mve (log size), bm (book-to-market), mom12m (12-month momentum) are
  cross-sectional z-scores; ret_fwd_1 = next month's return. Dates are month-ends 2021-12-31 .. 2025-10-31.
- sentiment_data.db / ravenpack: RavenPack news headlines 2023+, event_sentiment_score in [-1, 1], ret_fwd_1 of the stock.
  `test_sentiment_condition` aggregates the trailing 30 days per ticker into avg_sentiment and news_count.

## Questions to answer (from the research team)
How well did the system come to its scoring rule? Did it use tools, and were they helpful in generating the rule?
Score each rubric item 0-2 and back each score with evidence.

## Rubric
{rubric}

## Deterministic audit (computed by code from the log; trust these facts)
{json.dumps(a, indent=1, default=str)}

## Final output file (rulebook + zero-shot strategy)
{text}

## Full transcript
{render(traj)}
"""
    return user, meta


def judge(month: str, effort: str = "high") -> dict:
    user, meta = judge_prompt(month)
    run = run_claude(JUDGE_SYSTEM, user, judge_tools(meta), JUDGE_SCHEMA, B1 / month / "judge_log.jsonl", effort=effort)
    res = {"month": month, "judge_model": run.models_used[-1] if run.models_used else None, "usage": run.usage,
           "judge_sql_checks": len(run.tool_calls), **run.result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (B1 / month / "judge.json").write_text(json.dumps(res, indent=2))
    return res
