# Agentic benchmarks (1) and (2)

These benchmarks implement the two-stage benchmark plan of Nov 2026 (stages from Agostino's email of Nov 15).

| | Benchmark (1): decision process | Benchmark (2): performance explanation |
|---|---|---|
| System | `Llama 3 Agentic ACE` pipeline, run with **GPT-4o** (`gpt-4o-2024-08-06`) | **LLM-S ACE** (GPT-3.5) + residual nodewise / GMV, from `GPT 3.5 ACE/` |
| Inputs | Full chat history and tool log, prompts, data, deterministic audit | Screening rules, screened stocks, weights, return sequences, `firm_data.db` |
| LLM | Claude Opus 5.5 as **judge** (a different family from GPT-4o); blind to outcomes | Claude Opus 5.5 as **explainer**, with a read-only SQL tool |
| Checked by | Deterministic audit + 2 human reviewers per rubric item | Automatic recomputation of every typed claim + fact sheet + human score |
| Runs | 2 (Jul and Aug 2024 decision dates) | 1 |
| Notebook | `xilin/notebooks/benchmark1_process_scoring.ipynb` | `xilin/notebooks/benchmark2_performance_explanation.ipynb` |

## Files
- `common.py`: paths and keys (read from `<project>/<provider>_api_key.txt`, never committed), a read-only SQL tool, and `run_claude`, a logged tool loop with a JSON-schema final answer.
- `b1_ace_pipeline.py`: executes the preprint pipeline's notebook code (`paper_preprint_source()`) with three asserted edits (the LLM, the base URL, the rate limit) and records CrewAI's event bus (every LLM call, tool call and task) to JSONL.
- `b1_judge.py`: rubric P1–P9 (0/1/2 anchors), transcript renderer, deterministic audit (tool errors, look-ahead, use of forward returns, thresholds copied from prompt examples, whether the rule executes), the judge, and `outcome()`, which reveals returns only after scoring.
- `b2_attribution.py`: data assembly, the fact sheet (Sharpe decomposition, by year, leave-one-quarter-out, CAPM, stock / leg / industry contributions, exposures), the typed-claim schema, and `verify_claims`.
- `b2_explain.py`: runs the explainer and builds the review table.
- `results/`: shareable outputs (scores, claims, review sheets). No raw transcripts.

Raw transcripts, judge logs and the GPT-4o databases go to `data_local/benchmarks/` (git-ignored), because they contain licensed CRSP/Compustat and RavenPack data.

## Running on another machine

**1. Environment** (Python 3.11; from the Agentics repo root)
```
# macOS / Linux
python3.11 -m venv .venv-bench && source .venv-bench/bin/activate
# Windows (PowerShell)
py -3.11 -m venv .venv-bench; .venv-bench\Scripts\Activate.ps1

pip install -r xilin/benchmarks/requirements.txt
python -m ipykernel install --user --name agentic-screen --display-name "agentic-screen (Python 3.11)"
```
In VS Code, open the Agentics folder, install the Python and Jupyter extensions, open a notebook, and pick the
`agentic-screen` kernel (or the `.venv-bench` interpreter).

**2. Data (licensed, not in git).** Searched in this order: `$AGENTIC_DATA_DIR`, `<project>/data`, `../Data`
(the folder the preprint notebook uses), `../data`, then the repo root.

| File | Needed for |
|---|---|
| `green cleaned.csv` (full history) | benchmark (2): permno → ticker / industry, next-month returns. Usually in the repo root. |
| `green cleaned new.csv`, `ravenpack_sp500_permno_final.csv` | benchmark (1): `firm_data.db` / `sentiment_data.db` are built from these on first use (no API calls). |

The other inputs (`GPT 3.5 ACE/`, `Returns/`, `sp500_portfolio_weights.csv`, `Llama 3 Agentic ACE/`) are in the repo.

**3. The saved runs (to view without paying for new ones).** Raw transcripts contain licensed data, so they are shared
as a zip, not through git. Unzip `benchmark_runs_2026-11-19.zip` into `data_local/benchmarks/` so that
`data_local/benchmarks/b1/2024-07/trajectory.jsonl` exists. Without it:
- benchmark (2) still runs: it falls back to the committed `results/b2_explanation.json`;
- benchmark (1) stops with a message, because its notebook shows the transcripts themselves.

`data_local/` carries its own `.gitignore`, so nothing in it can be committed by accident.

**4. API keys (only for new runs).** With `RUN_PIPELINE` / `RUN_JUDGE` / `RUN_EXPLAINER = False` (the default), no
key is needed and nothing is charged. For new runs, set `OPENAI_API_KEY` (GPT-4o, benchmark 1) and
`ANTHROPIC_API_KEY` (Claude judge / explainer), or put `openai_api_key.txt` / `claude_api_key.txt` in the folder that
contains the repo. Approximate cost per run: GPT-4o ≈ $0.20 per month, judge ≈ $0.15–0.35 per month, explainer ≈ $0.35.

Xilin's Mac uses `.venv-mac` (created with `uv`) and keeps the data in `<project>/data`; nothing else differs.

## Data conventions (benchmark 2), verified against the preprint's output files
- A weight dated T earns the stock's excess return over month T+1. Σ w·r reproduces `Returns/LLM-S ACE 3.5.csv` (corr 0.9998); the gap is the 10 bp × turnover cost.
- The screen for year *y* is the eligible set for formation dates in calendar year *y* (100% overlap with the weights).
- Sharpe 0.8931 (ACE) uses ddof=1, and 0.2685 (S&P) uses ddof=0. On ddof=1 the S&P figure is 0.2642.
