"""
Run the BioReason pipeline against the benchmark suite, compute deterministic
grounding metrics + LLM-as-Judge scores, and persist a timestamped run artifact.

Usage:
    python -m eval.run_suite              # run all queries
    python -m eval.run_suite --limit 3    # smoke test on first 3 queries
    python -m eval.run_suite --ids q001,q003,q014   # specific queries
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Path setup so we can import project root modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from main import app  # compiled LangGraph workflow
from evaluator import grade_report
from eval.grounding import pmid_grounding_rate, numeric_grounding_rate

QUERIES_PATH = Path(__file__).parent / "benchmarks" / "queries.jsonl"
RUNS_DIR = Path(__file__).parent / "runs"


def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run_pipeline(query_text: str) -> tuple[str, str]:
    """Invoke the BioReason graph and extract (final_report, raw_tool_data)."""
    inputs = {"messages": [HumanMessage(content=query_text)]}
    final_state = app.invoke(inputs)
    final_report = final_state["messages"][-1].content
    raw_tool_data = "\n".join(
        msg.content for msg in final_state["messages"] if msg.type == "tool"
    )
    return final_report, raw_tool_data


def run_one_query(q: dict) -> dict:
    """Execute pipeline + judge + grounding checks for a single query."""
    qid = q["id"]
    query_text = q["query"]
    expected_pmids = q.get("expected_pmids", [])

    print(f"\n[{qid}] {query_text[:70]}...")
    t0 = time.time()

    try:
        final_report, raw_tool_data = run_pipeline(query_text)
    except Exception as e:
        print(f"  PIPELINE FAILED: {e}")
        return {
            "id": qid,
            "query": query_text,
            "category": q.get("category"),
            "status": "pipeline_failed",
            "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }

    elapsed = round(time.time() - t0, 2)

    # Deterministic grounding
    pmid_check = pmid_grounding_rate(final_report, raw_tool_data)
    numeric_check = numeric_grounding_rate(final_report, raw_tool_data)

    # LLM-as-Judge
    try:
        scorecard = grade_report(raw_tool_data, final_report)
        judge = {
            "hallucination_score": scorecard.hallucination_score,
            "precision_score": scorecard.precision_score,
            "passed": scorecard.passed,
            "reasoning": scorecard.evaluator_reasoning,
        }
    except Exception as e:
        judge = {"error": str(e)}

    result = {
        "id": qid,
        "query": query_text,
        "category": q.get("category"),
        "status": "ok",
        "elapsed_s": elapsed,
        "expected_pmids_count": len(expected_pmids),
        "final_report": final_report,
        "raw_tool_data_chars": len(raw_tool_data),
        "raw_tool_data_preview": raw_tool_data[:500],
        "pmid_grounding": pmid_check,
        "numeric_grounding": numeric_check,
        "judge": judge,
    }

    print(
        f"  PMID grounding: {pmid_check['grounded']}/{pmid_check['cited']} "
        f"(rate {pmid_check['rate']:.2f})  |  "
        f"Numeric grounding: {numeric_check['grounded']}/{numeric_check['cited']} "
        f"(rate {numeric_check['rate']:.2f})  |  "
        f"Judge H={judge.get('hallucination_score','-')}/10 "
        f"P={judge.get('precision_score','-')}/10  |  "
        f"{elapsed}s"
    )
    return result


def aggregate(results: list[dict]) -> dict:
    """Compute summary stats across all successful results."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"runs": len(results), "ok": 0}

    def avg(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    pmid_rates = [r["pmid_grounding"]["rate"] for r in ok]
    numeric_rates = [r["numeric_grounding"]["rate"] for r in ok]
    elapsed = [r["elapsed_s"] for r in ok]

    judge_h = [
        r["judge"]["hallucination_score"]
        for r in ok
        if isinstance(r["judge"].get("hallucination_score"), int)
    ]
    judge_p = [
        r["judge"]["precision_score"]
        for r in ok
        if isinstance(r["judge"].get("precision_score"), int)
    ]
    judge_pass = [r["judge"].get("passed") for r in ok if "passed" in r["judge"]]

    return {
        "runs": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "avg_pmid_grounding_rate": avg(pmid_rates),
        "avg_numeric_grounding_rate": avg(numeric_rates),
        "avg_judge_hallucination": avg(judge_h),
        "avg_judge_precision": avg(judge_p),
        "judge_pass_rate": avg([1 if p else 0 for p in judge_pass])
        if judge_pass
        else None,
        "avg_elapsed_s": avg(elapsed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries.")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated query IDs to run (e.g. q001,q003).")
    args = parser.parse_args()

    queries = load_queries()
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in wanted]
    elif args.limit:
        queries = queries[: args.limit]

    print(f"Running {len(queries)} queries.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for q in queries:
        results.append(run_one_query(q))

    summary = aggregate(results)

    with open(run_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to {run_dir}/results.json")


if __name__ == "__main__":
    main()