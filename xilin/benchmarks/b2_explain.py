"""Benchmark (2), step 2: ask an LLM to explain the outperformance, then verify its claims against the fact sheet."""
from __future__ import annotations

import json

from .b2_attribution import CLAIM_SCHEMA, EXPLAINER_SYSTEM, explainer_prompt, load, verify_claims
from .common import LOCAL, RESULTS, SQLTool, run_claude

B2 = LOCAL / "b2"


def explain(D: dict | None = None, effort: str = "high", tag: str = "run1") -> dict:
    D = D or load()
    tool = SQLTool(D["con"], "query_portfolio_data",
                   "Read-only DuckDB SQL over: portfolio_returns, ace_holdings, sp500_holdings, screens, strategies, firms "
                   "(see the table documentation in the question).")
    run = run_claude(EXPLAINER_SYSTEM, explainer_prompt(D), [tool], CLAIM_SCHEMA, B2 / f"explainer_{tag}_log.jsonl",
                     effort=effort)
    res = {"tag": tag, "model": run.models_used[-1] if run.models_used else None, "usage": run.usage,
           "sql_queries": len(run.tool_calls), **run.result}
    B2.mkdir(parents=True, exist_ok=True)
    (B2 / f"explainer_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def load_explanation(tag: str = "run1") -> dict:
    """The saved explainer run. Falls back to the committed copy in results/ when data_local has no transcript."""
    p = B2 / f"explainer_{tag}.json"
    if not p.exists() and tag == "run1":
        p = RESULTS / "b2_explanation.json"
    return json.loads(p.read_text())


def review_table(res: dict, D: dict):
    """Claims + automatic check, with empty columns for the two human reviewers (0/1/2)."""
    t = verify_claims(res["claims"], D)
    t["reviewer1_score_0_2"] = ""
    t["reviewer2_score_0_2"] = ""
    t["reviewer_notes"] = ""
    return t
