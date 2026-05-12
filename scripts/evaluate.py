"""scripts/evaluate.py
----------------------
End-to-end evaluation harness for the Ego User Modelling Agent.

Loads the held-out test_reviews from data/user_profiles.json, runs the
full user_modeling_agent pipeline on each, and reports three metrics:

  Metric          Target
  ──────────────  ──────
  ROUGE-L F1      > 0.35
  BERTScore F1    > 0.82
  Rating RMSE     < 0.80

Usage:
  PYTHONPATH=. python scripts/evaluate.py
  PYTHONPATH=. python scripts/evaluate.py --limit 20
  PYTHONPATH=. python scripts/evaluate.py --output results.json
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

# Add project root to sys.path so 'graphs' and 'core' can be imported easily
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR = Path("data")

TARGETS = {
    "rouge_l":       0.35,
    "bert_score_f1": 0.82,
    "rmse":          0.80,   # must be BELOW this threshold
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_cases(profiles_path: Path, limit: int | None = None) -> list[dict]:
    """
    Build a flat list of (user, item, ground_truth_rating, ground_truth_review)
    tuples sourced from each user's held-out test_reviews split.
    """
    with open(profiles_path, encoding="utf-8") as f:
        profiles: list[dict] = json.load(f)

    cases: list[dict] = []
    for profile in profiles:
        for review in profile.get("test_reviews", []):
            cases.append({
                "user_persona":        profile["name"],
                "user_id":             profile["user_id"],
                "item_metadata": {
                    "name":        review.get("product_name", "Unknown"),
                    "category":    review.get("category", "Unknown"),
                    "description": "",
                },
                "ground_truth_rating": float(review.get("rating", 3.0)),
                "ground_truth_review": (
                    (review.get("title", "") + " " + review.get("body", "")).strip()
                ),
            })
            if limit and len(cases) >= limit:
                return cases

    return cases


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_evaluation(cases: list[dict]) -> dict:
    """Invoke the agent on each test case and collect predictions."""
    # Import here so the module can be parsed without a live Qdrant connection.
    from graphs.task_a import user_modeling_agent

    predicted_ratings: list[float] = []
    actual_ratings:    list[float] = []
    predicted_reviews: list[str]   = []
    actual_reviews:    list[str]   = []
    errors = 0

    for i, case in enumerate(cases):
        log.info(
            "[%d/%d] user='%s'  item='%s'",
            i + 1, len(cases),
            case["user_persona"],
            case["item_metadata"]["name"],
        )
        try:
            result = user_modeling_agent.invoke({
                "user_persona":  case["user_persona"],
                "item_metadata": case["item_metadata"],
            })
            predicted_ratings.append(result["predicted_rating"])
            actual_ratings.append(case["ground_truth_rating"])
            predicted_reviews.append(
                result.get("final_review") or result.get("simulated_review", "")
            )
            actual_reviews.append(case["ground_truth_review"])
        except Exception as exc:
            log.warning("  ↳ Failed: %s", exc)
            errors += 1

    return {
        "predicted_ratings": predicted_ratings,
        "actual_ratings":    actual_ratings,
        "predicted_reviews": predicted_reviews,
        "actual_reviews":    actual_reviews,
        "errors":            errors,
        "total":             len(cases),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def calculate_rouge(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def calculate_bert_score(references: list[str], predictions: list[str]) -> float:
    _, _, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return float(F1.mean().item())


def calculate_rmse(actual: list[float], predicted: list[float]) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def compute_metrics(results: dict) -> dict:
    pred_r = results["predicted_ratings"]
    act_r  = results["actual_ratings"]
    pred_t = results["predicted_reviews"]
    act_t  = results["actual_reviews"]

    # Rating RMSE
    rmse = calculate_rmse(act_r, pred_r) if pred_r else float("inf")

    # ROUGE-L (per-pair average)
    rouge_scores = [
        calculate_rouge(ref, pred)
        for ref, pred in zip(act_t, pred_t)
        if ref and pred
    ]
    rouge_l = float(np.mean(rouge_scores)) if rouge_scores else 0.0

    # BERTScore F1
    bert_f1 = 0.0
    if pred_t and act_t:
        try:
            bert_f1 = calculate_bert_score(act_t, pred_t)
        except Exception as exc:
            log.warning("BERTScore computation failed: %s", exc)

    return {"rmse": rmse, "rouge_l": rouge_l, "bert_score_f1": bert_f1}


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(metrics: dict, results: dict) -> None:
    evaluated = results["total"] - results["errors"]
    print("\n" + "=" * 60)
    print("  EGO USER MODELLING AGENT — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Cases evaluated : {evaluated} / {results['total']}")
    print(f"  Errors          : {results['errors']}")
    print()

    checks = [
        ("ROUGE-L F1",    metrics["rouge_l"],       TARGETS["rouge_l"],       True),
        ("BERTScore F1",  metrics["bert_score_f1"], TARGETS["bert_score_f1"], True),
        ("Rating RMSE",   metrics["rmse"],           TARGETS["rmse"],          False),
    ]

    all_pass = True
    for name, value, target, higher_is_better in checks:
        passed = (value >= target) if higher_is_better else (value <= target)
        if not passed:
            all_pass = False
        status = "✅ PASS" if passed else "❌ FAIL"
        op = ">" if higher_is_better else "<"
        print(f"  {status}  {name:<16}: {value:.4f}  (target {op} {target})")

    print()
    print(f"  Overall: {'🎉 ALL TARGETS MET' if all_pass else '⚠️  TARGETS NOT MET'}")
    print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Ego user modelling agent against held-out test data."
    )
    parser.add_argument(
        "--profiles", type=Path,
        default=DATA_DIR / "user_profiles.json",
        help="Path to user_profiles.json (default: data/user_profiles.json).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of test cases to evaluate (default: all).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="If set, save full results dict to this JSON file.",
    )
    args = parser.parse_args()

    if not args.profiles.exists():
        log.error(
            "Profiles file not found at %s. "
            "Run scripts/build_user_profiles.py first.",
            args.profiles,
        )
        return

    log.info("Loading test cases from %s", args.profiles)
    cases = load_test_cases(args.profiles, limit=args.limit)
    log.info("Found %d test cases", len(cases))

    if not cases:
        log.error(
            "No test cases found. "
            "Ensure user profiles contain a 'test_reviews' key."
        )
        return

    t0 = time.time()
    results = run_evaluation(cases)
    elapsed = time.time() - t0
    log.info("Evaluation completed in %.1fs", elapsed)

    metrics = compute_metrics(results)
    print_report(metrics, results)

    if args.output:
        payload = {**metrics, "elapsed_seconds": elapsed, "summary": results}
        # Remove large lists from the saved summary to keep the file manageable
        payload["summary"].pop("predicted_reviews", None)
        payload["summary"].pop("actual_reviews", None)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
