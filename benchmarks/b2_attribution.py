"""Benchmark (2): why did LLM-S ACE + residual nodewise / GMV beat the S&P 500? An LLM explains; code checks.

Pieces
- `load()`             assembles the preprint's LLM-S ACE output files into one DuckDB (weights, returns, firms, screens, strategies).
- `fact_sheet()`       the deterministic attribution "truth": Sharpe decomposition, by-year results, CAPM alpha/beta,
                       stock / leg / industry contributions, characteristic exposures, leave-one-period-out Sharpe.
- `CLAIM_SCHEMA`       what the explainer must return: a narrative plus typed, machine-checkable claims.
- `verify_claims()`    recomputes every typed claim and marks it match / sign-only / mismatch / not auto-checkable.
                       A human then scores each claim 0-2 (see the notebook).

Conventions (stated to the explainer and used everywhere here)
- A weight dated T (month-end) earns that stock's return over month T+1 (`ret_fwd_1`, excess of the risk-free rate).
  `return_month` = the month in which the return is realized. Periods in claims refer to return months.
- The return files are net of 10 bp x turnover; per-stock contributions are gross (w x r), so they sum to the
  gross portfolio return, which differs from net by the cost drag.
- Sharpe = mean / std x sqrt(12). The reported 0.8931 uses the sample std (ddof=1); the reported 0.2685 for the S&P uses ddof=0.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import REPO, data_file

ACE_DIR = REPO / "GPT 3.5 ACE"
FILES = dict(
    ace_weights=ACE_DIR / "LLM-S ACE portfolio weights.csv",
    sp_weights=REPO / "sp500_portfolio_weights.csv",
    ace_returns=REPO / "Returns" / "LLM-S ACE 3.5.csv",
    sp_returns=REPO / "Returns" / "portfolio_returns_vw 3.5.csv",
    firm_db=ACE_DIR / "firm_data.db",
)
YEARS = (2021, 2022, 2023, 2024)


def _sr(x, ddof=1):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=ddof) * np.sqrt(12)) if len(x) > 2 and x.std(ddof=ddof) > 0 else np.nan


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def load() -> dict:
    """Returns a dict of DataFrames and a DuckDB connection `con` with the same tables (for the explainer's SQL tool)."""
    w = pd.read_csv(FILES["ace_weights"], parse_dates=["date"])
    s = pd.read_csv(FILES["sp_weights"], parse_dates=["date"])
    dates = sorted(w.date.unique())
    rets = pd.DataFrame({"formation_date": dates,
                         "return_month": [(pd.Timestamp(d) + pd.offsets.MonthEnd(1)) for d in dates],
                         "ace": pd.read_csv(FILES["ace_returns"]).iloc[:, 0].values,
                         "sp500": pd.read_csv(FILES["sp_returns"]).iloc[:, 0].values})
    assert len(rets) == len(dates) == 31

    # permno -> ticker / name / industry, and next-month excess return, from the full green file (permno-keyed)
    g = pd.read_csv(data_file("green cleaned.csv"), usecols=["datadate", "permno", "ticker", "comnam", "industry", "sic2",
                                             "ret_excess", "bm", "mve", "mom12m"], parse_dates=["datadate"])
    g = g.sort_values(["permno", "datadate"])
    g["ret_fwd_1"] = g.groupby("permno").ret_excess.shift(-1)
    panel = g[g.datadate.isin(dates)].rename(columns={"datadate": "formation_date"})

    def holdings(W, name):
        h = W.rename(columns={"date": "formation_date"}).merge(panel, on=["formation_date", "permno"], how="left")
        h["return_month"] = h.formation_date + pd.offsets.MonthEnd(1)
        h["contribution"] = h.weight * h.ret_fwd_1
        h["portfolio"] = name
        return h

    ace, sp = holdings(w, "ace"), holdings(s, "sp500")
    assert ace.ret_fwd_1.notna().all() and sp.ret_fwd_1.notna().all()

    screens = []
    for y in YEARS:
        for lab in ("buys", "sells"):
            f = pd.read_csv(ACE_DIR / f"{lab}_{y}.csv", parse_dates=["datadate"])
            screens.append(f.assign(screen_year=y, label="BUY" if lab == "buys" else "SELL"))
    screens = pd.concat(screens)[["screen_year", "datadate", "permno", "ticker", "comnam", "label"]]
    strategies = pd.DataFrame([{"screen_year": y, "text": (ACE_DIR / f"systematic_strategy_{y}.txt").read_text()}
                               for y in YEARS])

    # label each holding with the screen that produced it. Checked against the weights file: every name held at a
    # formation date in calendar year y is in screen y (100% overlap), so screen_year = formation year.
    ace["screen_year"] = ace.formation_date.dt.year
    ace = ace.merge(screens[["screen_year", "permno", "label"]].drop_duplicates(["screen_year", "permno"]),
                    on=["screen_year", "permno"], how="left")

    firms = pd.read_sql_query("SELECT * FROM firms", sqlite3.connect(FILES["firm_db"]))

    con = duckdb.connect()
    for name, df in dict(ace_holdings=ace.drop(columns="portfolio"), sp500_holdings=sp.drop(columns="portfolio"),
                         portfolio_returns=rets, screens=screens, strategies=strategies, firms=firms).items():
        con.register(f"_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    return dict(ace=ace, sp=sp, rets=rets, screens=screens, strategies=strategies, firms=firms, con=con)


TABLE_DOCS = """
Tables (DuckDB SQL):
- portfolio_returns(formation_date, return_month, ace, sp500): 31 monthly NET excess returns (decimal; 0.01 = 1%).
  `ace` = LLM-S ACE screen + residual nodewise precision matrix + GMV weights (10 bp x turnover costs deducted).
  `sp500` = value-weighted S&P 500. The row with formation_date T is the return realized over month T+1 (= return_month).
- ace_holdings(formation_date, permno, weight, ticker, comnam, industry, sic2, ret_excess, bm, mve, mom12m, ret_fwd_1,
  return_month, contribution, screen_year, label): one row per stock per month in the ACE portfolio.
  weight: portfolio weight set at formation_date (sums to 1 each month; negative = short).
  ret_fwd_1: the stock's excess return over return_month. contribution = weight * ret_fwd_1 (gross, before costs).
  bm, mve, mom12m: cross-sectionally standardized z-scores of book-to-market, log size, 12-month momentum at formation_date.
  label: BUY or SELL = the LLM-S ACE screen label that put the stock in the eligible set (screen_year = the screen in force).
  ret_excess: the stock's excess return in the formation month itself (already realized at formation).
- sp500_holdings(...same columns minus screen_year/label...): the value-weighted S&P 500 portfolio, same conventions.
- screens(screen_year, datadate, permno, ticker, comnam, label): the full LLM-S ACE BUY/SELL lists. The screen for
  year y is formed from characteristics at `datadate` and is the eligible set for formation dates in calendar year y.
  Only screened names with a complete 180-month return history can receive weight.
- strategies(screen_year, text): the LLM-S ACE rulebook and final zero-shot rule that produced each year's screen.
- firms(datadate, ticker, ret_excess, bm, mve, mom12m, ret_fwd_1): the pipeline's firm_data.db, all S&P 500 stocks 1980-2024.
"""


# ---------------------------------------------------------------------------------------------------------------
# Deterministic attribution ("truth")
# ---------------------------------------------------------------------------------------------------------------
def perf(x: pd.Series) -> dict:
    x = pd.Series(np.asarray(x, float))
    cum = (1 + x).cumprod()
    return dict(months=len(x), monthly_mean=x.mean(), monthly_vol=x.std(ddof=1), ann_mean=x.mean() * 12,
                ann_vol=x.std(ddof=1) * np.sqrt(12), sharpe=_sr(x, 1), sharpe_ddof0=_sr(x, 0),
                cumulative_return=cum.iloc[-1] - 1, max_drawdown=float((cum / cum.cummax() - 1).min()),
                hit_rate=float((x > 0).mean()))


def capm(y, x) -> dict:
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.c_[np.ones_like(x), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return dict(alpha_monthly=b[0], alpha_ann=b[0] * 12, alpha_t=b[0] / se[0], beta=b[1], beta_t=b[1] / se[1],
                correlation=float(np.corrcoef(y, x)[0, 1]), r2=1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))


def _in(df, start=None, end=None, col="return_month"):
    m = pd.Series(True, index=df.index)
    if start:
        m &= df[col] >= pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    if end:
        m &= df[col] <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    return df[m]


def fact_sheet(D: dict) -> dict[str, pd.DataFrame]:
    rets, ace, sp = D["rets"], D["ace"], D["sp"]
    F = {}
    gross = ace.groupby("return_month").contribution.sum()
    F["headline"] = pd.DataFrame({"ace (net)": perf(rets.ace), "sp500 (net)": perf(rets.sp500),
                                  "ace gross (sum w*r)": perf(gross.values)}).T

    a, b = perf(rets.ace), perf(rets.sp500)
    F["sharpe_gap"] = pd.DataFrame([{
        "sharpe_ace": a["sharpe"], "sharpe_sp500 (ddof=1)": b["sharpe"], "sharpe_sp500 (ddof=0, reported 0.2685)": b["sharpe_ddof0"],
        "gap (ddof=1)": a["sharpe"] - b["sharpe"],
        "from higher mean": (a["monthly_mean"] - b["monthly_mean"]) / b["monthly_vol"] * np.sqrt(12),
        "from lower vol": a["monthly_mean"] * (1 / a["monthly_vol"] - 1 / b["monthly_vol"]) * np.sqrt(12),
        "ann mean ace - sp": a["ann_mean"] - b["ann_mean"], "ann vol ace - sp": a["ann_vol"] - b["ann_vol"],
        "cost drag (ann, gross-net)": (gross.values.mean() - rets.ace.mean()) * 12}]).T.rename(columns={0: "value"})

    F["capm_ace_on_sp500"] = pd.DataFrame([capm(rets.ace, rets.sp500)]).T.rename(columns={0: "value"})

    r = rets.assign(year=rets.return_month.dt.year)
    rows = []
    for y, g in r.groupby("year"):
        pa, pb = perf(g.ace), perf(g.sp500)
        rows.append(dict(year=y, months=len(g), ace_ann_mean=pa["ann_mean"], sp_ann_mean=pb["ann_mean"],
                         ace_ann_vol=pa["ann_vol"], sp_ann_vol=pb["ann_vol"], ace_sharpe=pa["sharpe"], sp_sharpe=pb["sharpe"],
                         ace_cum=pa["cumulative_return"], sp_cum=pb["cumulative_return"],
                         ace_sharpe_excl_year=_sr(r[r.year != y].ace), sp_sharpe_excl_year=_sr(r[r.year != y].sp500)))
    F["by_year"] = pd.DataFrame(rows).set_index("year")

    r["q"] = r.return_month.dt.to_period("Q").astype(str)
    F["leave_one_quarter_out"] = pd.DataFrame([dict(quarter=q, ace_sharpe_excl=_sr(r[r.q != q].ace),
                                                    sp_sharpe_excl=_sr(r[r.q != q].sp500),
                                                    ace_quarter_cum=(1 + g.ace).prod() - 1, sp_quarter_cum=(1 + g.sp500).prod() - 1)
                                               for q, g in r.groupby("q")]).set_index("quarter")
    F["best_worst_months"] = pd.concat([r.nlargest(5, "ace"), r.nsmallest(5, "ace")])[["return_month", "ace", "sp500"]]

    tot = ace.contribution.sum()
    st = ace.groupby(["ticker", "comnam"]).agg(months_held=("weight", "size"), avg_weight=("weight", "mean"),
                                               sum_contribution=("contribution", "sum")).reset_index()
    st["share_of_total"] = st.sum_contribution / tot
    st["ace_sharpe_without_stock"] = [
        _sr(rets.ace.values - ace[ace.ticker == t].groupby("return_month").contribution.sum()
            .reindex(rets.return_month, fill_value=0).values) for t in st.ticker]
    st = st.sort_values("sum_contribution", ascending=False)
    F["top_stocks"], F["bottom_stocks"] = st.head(12), st.tail(8).iloc[::-1]
    F["_stocks_all"] = st

    yr = ace.assign(year=ace.return_month.dt.year).groupby(["year", "ticker"]).contribution.sum().reset_index()
    yr["share_of_year"] = yr.contribution / yr.groupby("year").contribution.transform("sum")
    F["top_stock_each_year"] = yr.sort_values("contribution", ascending=False).groupby("year").head(3).sort_values(
        ["year", "contribution"], ascending=[True, False])

    leg = ace.assign(side=np.where(ace.weight > 0, "long", "short"))
    F["legs"] = leg.groupby(["label", "side"]).agg(names=("ticker", "nunique"), avg_weight=("weight", "mean"),
                                                   sum_contribution=("contribution", "sum"),
                                                   hit_rate=("ret_fwd_1", lambda r_: np.nan)).drop(columns="hit_rate")
    F["legs"]["share_of_total"] = F["legs"].sum_contribution / tot
    F["legs_by_sign"] = leg.groupby("side").contribution.sum().to_frame("sum_contribution").assign(share_of_total=lambda d: d.sum_contribution / tot)

    # industry: average net weight vs S&P, contribution, and Brinson-Fachler allocation/selection vs S&P (gross)
    def ind_month(h):
        return h.groupby(["return_month", "industry"]).agg(w=("weight", "sum"), c=("contribution", "sum")).reset_index()
    ia, ib = ind_month(ace), ind_month(sp)
    m = ia.merge(ib, on=["return_month", "industry"], how="outer", suffixes=("_p", "_b")).fillna(0)
    m["r_b"] = np.where(m.w_b != 0, m.c_b / m.w_b.replace(0, np.nan), 0)
    m["r_p"] = np.where(m.w_p != 0, m.c_p / m.w_p.replace(0, np.nan), 0)
    Rb = m.groupby("return_month").c_b.transform("sum")
    m["allocation"] = (m.w_p - m.w_b) * (m.r_b - Rb)
    m["selection"] = np.where(m.w_b != 0, m.w_p * (m.r_p - m.r_b), m.c_p - m.w_p * Rb)  # no S&P weight: all is selection
    F["industry"] = m.groupby("industry").agg(avg_net_weight_ace=("w_p", "mean"), avg_weight_sp500=("w_b", "mean"),
                                              contribution_ace=("c_p", "sum"), allocation=("allocation", "sum"),
                                              selection=("selection", "sum")).assign(
        active_weight=lambda d: d.avg_net_weight_ace - d.avg_weight_sp500).sort_values("contribution_ace", ascending=False)

    def expo(h):
        return h.groupby("return_month").apply(lambda g: pd.Series({c: (g.weight * g[c]).sum() for c in ("bm", "mve", "mom12m")}))
    ea, eb = expo(ace), expo(sp)
    F["characteristic_exposure"] = pd.DataFrame({"ace (net-weighted z)": ea.mean(), "sp500 (weighted z)": eb.mean(),
                                                 "active": ea.mean() - eb.mean()})
    F["portfolio_shape_by_year"] = ace.assign(year=ace.return_month.dt.year).groupby("year").apply(
        lambda g: pd.Series({"avg_names": g.groupby("formation_date").size().mean(),
                             "avg_gross_leverage": g.groupby("formation_date").weight.apply(lambda x: x.abs().sum()).mean(),
                             "avg_short_weight": g[g.weight < 0].groupby("formation_date").weight.sum().mean(),
                             "share_names_labelled_SELL": (g.label == "SELL").mean()}))
    return F


# ---------------------------------------------------------------------------------------------------------------
# Claims: schema for the explainer, and the automatic verifier
# ---------------------------------------------------------------------------------------------------------------
METRICS = ["sharpe", "ann_mean", "ann_vol", "monthly_mean", "monthly_vol", "cumulative_return", "max_drawdown",
           "hit_rate", "alpha_ann", "beta", "correlation", "sharpe_excluding_period",
           "sum_contribution", "share_of_total", "avg_weight", "avg_net_weight", "active_weight",
           "weighted_avg_z", "none"]
KINDS = ["portfolio_metric", "stock_contribution", "leg_contribution", "industry", "characteristic_exposure",
         "screening_rule", "weighting", "qualitative"]


def _nullable(t):
    return {"anyOf": [{"type": t}, {"type": "null"}]}


CLAIM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "explanation", "claims", "caveats"],
    "properties": {
        "summary": {"type": "string", "description": "3-5 sentence answer to the question."},
        "explanation": {"type": "string", "description": "Detailed explanation in markdown, referencing claim ids like [C3]."},
        "caveats": {"type": "string", "description": "What you could not establish, and why."},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "text", "kind", "portfolio", "metric", "entity", "period_start", "period_end", "value", "evidence"],
            "properties": {
                "id": {"type": "string", "description": "C1, C2, ..."},
                "text": {"type": "string", "description": "The claim in one sentence."},
                "kind": {"type": "string", "enum": KINDS},
                "portfolio": {"type": "string", "enum": ["ace", "sp500", "active", "none"],
                              "description": "active = the active return series ace minus sp500 (e.g. hit_rate = share of months ace beat sp500)"},
                "metric": {"type": "string", "enum": METRICS},
                "entity": _nullable("string") | {"description": "ticker (stock_contribution), long|short|BUY|SELL|BUY_long|BUY_short|SELL_long|SELL_short (leg_contribution), industry name (industry), bm|mve|mom12m (characteristic_exposure); else null"},
                "period_start": _nullable("string") | {"description": "first return month YYYY-MM (inclusive), null = whole sample"},
                "period_end": _nullable("string") | {"description": "last return month YYYY-MM (inclusive), null = whole sample"},
                "value": _nullable("number") | {"description": "the number claimed, in decimals (0.05 = 5%); Sharpe annualized; null if qualitative"},
                "evidence": {"type": "string", "description": "the SQL query/result the claim rests on, or 'reasoning' / 'prior knowledge'"},
            }}},
    },
}

ABS_TOL = dict(sharpe=0.10, sharpe_excluding_period=0.10, ann_mean=0.01, ann_vol=0.01, monthly_mean=0.001, monthly_vol=0.001,
               cumulative_return=0.01, max_drawdown=0.01, hit_rate=0.03, alpha_ann=0.01, beta=0.05, correlation=0.05,
               sum_contribution=0.005, share_of_total=0.03, avg_weight=0.003, avg_net_weight=0.005, active_weight=0.005,
               weighted_avg_z=0.05)


def _month(s):
    return None if s in (None, "") else pd.Period(s[:7], "M").to_timestamp("M")


def truth_for(c: dict, D: dict) -> float | None:
    """Recompute the quantity a typed claim refers to. Returns None if the claim is not auto-checkable."""
    k, met, ent, port = c["kind"], c["metric"], c.get("entity"), c["portfolio"]
    a, b = _month(c.get("period_start")), _month(c.get("period_end"))
    rets, ace, sp = D["rets"], D["ace"], D["sp"]
    R = _in(rets, a, b)
    def series(df):   # "active" = the active return series ace - sp500 (so hit_rate = share of months ACE beat the index)
        return df.ace - df.sp500 if port == "active" else df[port if port in ("ace", "sp500") else "ace"]
    if k == "portfolio_metric":
        if met == "sharpe_excluding_period":
            return _sr(series(rets[~rets.index.isin(R.index)]))
        if met in ("alpha_ann", "beta", "correlation"):
            return capm(R.ace, R.sp500)[met]
        if met in perf(rets.ace):
            return perf(series(R))[met]
    H = _in(ace, a, b)
    tot = H.contribution.sum()
    months = sorted(H.return_month.unique())
    LEGS = {"LONG", "SHORT", "BUY", "SELL", "BUY_LONG", "BUY_SHORT", "SELL_LONG", "SELL_SHORT"}
    if k == "stock_contribution" and ent:
        tickers = [t.strip().upper() for t in re.split(r"[,+/&]|\band\b", ent) if t.strip()]
        h = H[H.ticker.str.upper().isin(tickers)]
        if h.empty or not set(tickers) <= set(H.ticker.str.upper()):
            return np.nan                                   # a stock never held in that period: claim is wrong
        return {"sum_contribution": h.contribution.sum(), "share_of_total": h.contribution.sum() / tot,
                "avg_weight": h.weight.mean()}.get(met)
    if k in ("leg_contribution", "weighting") and ent and ent.upper() in LEGS:
        e = ent.upper()
        mask = pd.Series(True, index=H.index)
        if "LONG" in e:
            mask &= H.weight > 0
        if "SHORT" in e:
            mask &= H.weight < 0
        if e.startswith("BUY"):
            mask &= H.label == "BUY"
        if e.startswith("SELL"):
            mask &= H.label == "SELL"
        h = H[mask]
        return {"sum_contribution": h.contribution.sum(), "share_of_total": h.contribution.sum() / tot,
                "avg_weight": h.weight.mean(),
                "avg_net_weight": h.groupby("return_month").weight.sum().reindex(months, fill_value=0).mean()}.get(met)
    if k == "industry" and ent:
        P = H if port != "sp500" else _in(sp, a, b)
        g = P[P.industry.str.lower() == ent.lower()].groupby("return_month").agg(w=("weight", "sum"), c=("contribution", "sum"))
        g = g.reindex(sorted(P.return_month.unique()), fill_value=0)
        if met in ("avg_net_weight", "avg_weight"):
            return g.w.mean()
        if met == "active_weight":
            s = _in(sp, a, b)
            sw = s[s.industry.str.lower() == ent.lower()].groupby("return_month").weight.sum().reindex(g.index, fill_value=0)
            return g.w.mean() - sw.mean()
        if met == "sum_contribution":
            return g.c.sum()
        if met == "share_of_total":
            return g.c.sum() / tot
    if k == "characteristic_exposure" and ent in ("bm", "mve", "mom12m") and met == "weighted_avg_z":
        def ex(P):
            return P.groupby("return_month").apply(lambda g: (g.weight * g[ent]).sum()).mean()
        if port == "sp500":
            return ex(_in(sp, a, b))
        return ex(H) - (ex(_in(sp, a, b)) if port == "active" else 0)
    return None


def verify_claims(claims: list[dict], D: dict) -> pd.DataFrame:
    rows = []
    for c in claims:
        t = truth_for(c, D) if c.get("value") is not None and c["metric"] != "none" else None
        v = c.get("value")
        if t is None:
            status = "not auto-checkable"
        elif t is not None and np.isnan(t):
            status = "mismatch (entity not held in period)"
        else:
            tol = max(ABS_TOL.get(c["metric"], 0.01), 0.10 * abs(t))
            status = "match" if abs(v - t) <= tol else ("sign/rank only" if np.sign(v) == np.sign(t) else "mismatch")
        rows.append({"id": c["id"], "kind": c["kind"], "claim": c["text"], "metric": c["metric"], "entity": c.get("entity"),
                     "period": f"{c.get('period_start') or 'start'}..{c.get('period_end') or 'end'}",
                     "claimed": v, "recomputed": t, "auto_check": status, "evidence": c.get("evidence", "")})
    return pd.DataFrame(rows)


EXPLAINER_SYSTEM = """You are a quantitative portfolio analyst writing a performance attribution for a research paper.
You have a read-only SQL tool over the portfolio data. Rules:
- Every number you state must come from a query you ran in this session. Do not rely on memory of market history;
  if you use prior knowledge (e.g. what happened to an industry in 2022), label it as such in `evidence`.
- Separate what the data shows from conjecture. Put conjecture in the explanation, not in quantitative claims.
- Prefer a small number of precise, checkable claims over many vague ones. Each claim must be typed so a script
  can recompute it: give the portfolio, metric, entity, period (return months, YYYY-MM) and value in decimals.
- Aggregate in SQL; query results are truncated to 60 rows.
- Also say how the screening rule and the weighting step each contributed, and anything surprising
  (for example, positions whose sign disagrees with the screen label, or inconsistencies in the stated numbers)."""


def explainer_prompt(D: dict) -> str:
    strat = "\n\n".join(f"===== screen_year {r.screen_year} =====\n{r.text}" for r in D["strategies"].itertuples())
    counts = D["screens"].groupby(["screen_year", "label"]).size().unstack(fill_value=0).to_string()
    return f"""Question (from the research team):
LLM-S ACE (GPT-3.5 with Agentic Context Engineering) writes a yearly BUY/SELL screening rule on three standardized
characteristics (bm, mve, mom12m). The BUY and SELL names form the eligible set; a residual nodewise precision-matrix
estimator with a global-minimum-variance objective (180-month window, monthly rebalancing, weights unconstrained in
sign) then sets the weights. Out of sample, Nov 2021 - May 2024 (31 return months), this portfolio has an annualized
Sharpe ratio of 0.8931, versus 0.2685 for the value-weighted S&P 500.

Explain in detail WHY, with these screening rules and these weights, the portfolio outperformed the S&P 500.
Decompose the Sharpe gap, identify what drove it (periods, stocks, long vs short legs, BUY vs SELL screens,
industries, characteristic exposures, the weighting step), and state each finding as a typed, checkable claim.

{TABLE_DOCS}
Screen sizes (number of names per screen):
{counts}

The LLM-S ACE outputs (rulebook + final rule) for each screen year:
{strat}
"""
