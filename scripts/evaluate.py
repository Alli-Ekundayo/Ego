"""scripts/evaluate.py
----------------------
End-to-end evaluation harness for the Ego Agent (Task A & Task B).

Usage:
  PYTHONPATH=. python scripts/evaluate.py --task a --limit 10
  PYTHONPATH=. python scripts/evaluate.py --task b --limit 10
  PYTHONPATH=. python scripts/evaluate.py --task both
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR = Path("data")

TARGETS = {
    "rouge_l": 0.35,
    "bert_score_f1": 0.82,
    "rmse": 0.80,
    "ndcg": 0.15,
}


def calculate_rouge(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def calculate_bert_score(references: list[str], predictions: list[str]) -> float:
    try:
        _, _, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        return float(F1.mean().item())
    except Exception as exc:
        log.warning("BERTScore computation failed: %s", exc)
        return 0.0


def calculate_rmse(actual: list[float], predicted: list[float]) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def calculate_ndcg(
    actual_item_ids: set[str], predicted_items: list[dict], k: int = 10
) -> float:
    """
    Calculate NDCG@k for a single user's recommendations.
    """
    if not predicted_items:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(predicted_items[:k]):
        item_id = str(item.get("item_id", ""))
        if item_id in actual_item_ids:
            dcg += 1.0 / np.log2(i + 2)

    idcg = 0.0
    num_relevant = min(len(actual_item_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def load_task_a_cases(profiles_path: Path, limit: int | None = None) -> list[dict]:
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    cases = []
    max_test_reviews = (
        max(len(p.get("test_reviews", [])) for p in profiles) if profiles else 0
    )
    for idx in range(max_test_reviews):
        for profile in profiles:
            test_reviews = profile.get("test_reviews", [])
            if idx < len(test_reviews):
                review = test_reviews[idx]
                cases.append(
                    {
                        "user_persona": profile["name"],
                        "item_metadata": {
                            "name": review.get("product_name", "Unknown"),
                            "category": review.get("category", "Unknown"),
                            "description": "",
                        },
                        "ground_truth_rating": float(review.get("rating", 3.0)),
                        "ground_truth_review": (
                            review.get("title", "") + " " + review.get("body", "")
                        ).strip(),
                    }
                )
                if limit and len(cases) >= limit:
                    return cases
    return cases



def evaluate_task_a(cases: list[dict]) -> dict:
    from graphs.task_a import user_modeling_agent

    pred_r, act_r, pred_t, act_t = [], [], [], []
    errors = 0
    for i, case in enumerate(cases):
        log.info("[%d/%d] Task A: user='%s'", i + 1, len(cases), case["user_persona"])
        try:
            res = user_modeling_agent.invoke(
                {
                    "user_persona": case["user_persona"],
                    "item_metadata": case["item_metadata"],
                }
            )
            pred_r.append(res["predicted_rating"])
            act_r.append(case["ground_truth_rating"])
            pred_t.append(res.get("final_review") or res.get("simulated_review", ""))
            act_t.append(case["ground_truth_review"])
        except Exception as exc:
            log.warning("  ↳ Failed: %s", exc)
            traceback.print_exc()
            errors += 1

    rmse = calculate_rmse(act_r, pred_r) if pred_r else float("inf")
    rouge_scores = [calculate_rouge(r, p) for r, p in zip(act_t, pred_t) if r and p]
    rouge_l = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    bert_f1 = calculate_bert_score(act_t, pred_t) if act_t else 0.0

    return {
        "rmse": rmse,
        "rouge_l": rouge_l,
        "bert_score_f1": bert_f1,
        "errors": errors,
    }


def load_task_b_cases(profiles_path: Path, limit: int | None = None) -> list[dict]:
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    cases = []
    for profile in profiles:
        test_items = {
            str(r.get("product_id", ""))
            for r in profile.get("test_reviews", [])
            if r.get("product_id")
        }
        if not test_items:
            continue
        cases.append(
            {
                "user_id": profile["user_id"],
                "user_persona": profile["name"],
                "test_item_ids": test_items,
                "context": "Recommend some products based on my interests",
            }
        )
        if limit and len(cases) >= limit:
            break
    return cases


def evaluate_task_b(cases: list[dict]) -> dict:
    from graphs.task_b import task_b_graph

    ndcg_scores = []
    errors = 0
    for i, case in enumerate(cases):
        log.info("[%d/%d] Task B: user='%s'", i + 1, len(cases), case["user_persona"])
        try:
            initial_state = {
                "user_id": case["user_id"],
                "context_text": case["context"],
                "persona_description": case["user_persona"],
                "session_history": [],
                "n": 10,
                "domain_filter": None,
                "new_product_features": {},
                "profile": None,
                "is_cold_start": False,
                "structured_signals": {},
                "extracted_domain": None,
                "extracted_aspects": [],
                "aspect_queries": [],
                "sparse_keywords": [],
                "proxy_embedding": [],
                "candidates": [],
                "ranked_recommendations": [],
                "refined_context_text": case["context"],
                "error": None,
            }
            res = task_b_graph.invoke(initial_state)

            if isinstance(res, list):
                res = res[-1] if res else {}

            recs = res.get("ranked_recommendations") or res.get("recommendations") or []

            if recs and isinstance(recs[0], list):
                recs = [item for sublist in recs for item in sublist]

            log.info(
                "  ↳ Recs: %d, Ground Truth: %d", len(recs), len(case["test_item_ids"])
            )
            rec_ids = [str(r.get("item_id")) for r in recs]
            gt_ids = [str(gid) for gid in case["test_item_ids"]]
            log.info("  ↳ Rec IDs: %s", rec_ids)
            log.info("  ↳ GT IDs: %s", gt_ids)

            hits = [rid for rid in rec_ids if rid in gt_ids]
            if hits:
                log.info("  ↳ HITS FOUND: %s", hits)
            else:
                log.info("  ↳ No hits in top %d", len(recs))

            score = calculate_ndcg(case["test_item_ids"], recs, k=10)
            ndcg_scores.append(score)
        except Exception as exc:
            log.warning("  ↳ Failed: %s", exc)
            traceback.print_exc()
            errors += 1

    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return {"ndcg": ndcg, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["a", "b", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    profiles_path = DATA_DIR / "user_profiles.json"
    results = {}

    if args.task in ["a", "both"]:
        log.info("Starting Task A Evaluation...")
        cases = load_task_a_cases(profiles_path, limit=args.limit)
        results["task_a"] = evaluate_task_a(cases)

    if args.task in ["b", "both"]:
        log.info("Starting Task B Evaluation...")
        cases = load_task_b_cases(profiles_path, limit=args.limit)
        results["task_b"] = evaluate_task_b(cases)

    print("\n" + "=" * 60 + "\n  EGO EVALUATION REPORT\n" + "=" * 60)
    if "task_a" in results:
        m = results["task_a"]
        print("Task A (Modelling):")
        print(
            f"  {'✅' if m['rouge_l'] >= TARGETS['rouge_l'] else '❌'} ROUGE-L: {m['rouge_l']:.4f} (target > {TARGETS['rouge_l']})"
        )
        print(
            f"  {'✅' if m['bert_score_f1'] >= TARGETS['bert_score_f1'] else '❌'} BERTScore: {m['bert_score_f1']:.4f} (target > {TARGETS['bert_score_f1']})"
        )
        print(
            f"  {'✅' if m['rmse'] <= TARGETS['rmse'] else '❌'} RMSE: {m['rmse']:.4f} (target < {TARGETS['rmse']})"
        )

    if "task_b" in results:
        m = results["task_b"]
        print("\nTask B (Recommendation):")
        print(
            f"  {'✅' if m['ndcg'] >= TARGETS['ndcg'] else '❌'} NDCG@10: {m['ndcg']:.4f} (target > {TARGETS['ndcg']})"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
