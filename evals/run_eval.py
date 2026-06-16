"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Calls the agent, then reconstructs correctness at every generate/revise
    step from the agent's `history` (each entry carries the SQL produced at
    that step). Running each step's SQL and comparing to the gold rows tells
    us "would this have passed if the agent stopped at iteration k".
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Reference rows: run the gold query once.
    _gold_ok, gold_rows, _gold_err = run_sql(db_id, gold_sql)

    # Call the agent over HTTP.
    payload = {"question": question["question"], "db": db_id}
    try:
        resp = httpx.post(agent_url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        # Agent failed entirely: no iterations, counts as incorrect.
        return {
            "db_id": db_id,
            "question": question["question"],
            "gold_sql": gold_sql,
            "final_sql": "",
            "num_iterations": 0,
            "iter_correct": [],
            "final_correct": False,
            "agent_ok": False,
            "agent_error": f"{type(e).__name__}: {e}",
        }

    history = data.get("history", [])

    # Reconstruct correctness at each generate/revise step.
    iter_correct: list[bool] = []
    for step in history:
        _ok, pred_rows, _err = run_sql(db_id, step.get("sql", ""))
        iter_correct.append(matches(gold_rows, pred_rows))

    # The agent's final answer is the last step it produced.
    final_correct = iter_correct[-1] if iter_correct else False

    return {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "final_sql": data.get("sql", ""),
        "num_iterations": len(history),
        "iter_correct": iter_correct,
        "final_correct": final_correct,
        "agent_ok": data.get("ok", False),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    overall = sum(1 for r in results if r["final_correct"]) / n
    avg_iters = sum(r["num_iterations"] for r in results) / n

    # How many iteration columns to report (the longest run seen).
    max_iters = max((len(r["iter_correct"]) for r in results), default=0)

    pass_rate_by_iteration: dict[str, float] = {}
    for k in range(max_iters):
        correct = 0
        for r in results:
            ic = r["iter_correct"]
            if not ic:
                val = False                 # agent failed outright
            elif k < len(ic):
                val = ic[k]                 # has a result at iteration k
            else:
                val = ic[-1]                # carry forward: it stopped earlier
            correct += int(bool(val))
        pass_rate_by_iteration[f"iter_{k}"] = correct / n

    return {
        "n": n,
        "overall_pass_rate": round(overall, 4),
        "avg_iterations": round(avg_iters, 4),
        "pass_rate_by_iteration": {k: round(v, 4) for k, v in pass_rate_by_iteration.items()},
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
