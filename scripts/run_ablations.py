import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import main as evaluate_main

logging.basicConfig(level=logging.INFO)

def run_ablation(name, patch_target, patch_func, task, limit):
    print(f"\n{'='*60}\nRunning Ablation: {name}\n{'='*60}")
    sys.argv = ["evaluate.py", "--task", task]
    if limit:
        sys.argv.extend(["--limit", str(limit)])
    
    with patch(patch_target, side_effect=patch_func) as p:
        try:
            evaluate_main()
        except Exception as e:
            print(f"Error during ablation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    print(f"\n{'='*60}\nRunning Baseline Task A\n{'='*60}")
    sys.argv = ["evaluate.py", "--task", "a", "--limit", str(args.limit)]
    evaluate_main()

    def patch_jumia(*args, **kwargs):
        class MockResult:
            payload = {"text": "I like this product."}
        return []

    run_ablation("No Jumia Context (Task A)", "core.vector_store.VectorStore.search", patch_jumia, "a", args.limit)

    print(f"\n{'='*60}\nRunning Baseline Task B\n{'='*60}")
    sys.argv = ["evaluate.py", "--task", "b", "--limit", str(args.limit)]
    evaluate_main()

    def patch_hybrid(dense_results, sparse_keywords, top_k=100, **kwargs):
        return dense_results[:top_k]
    run_ablation("No Sparse Retrieval BM25 (Task B)", "core.hybrid_search.hybrid_search", patch_hybrid, "b", args.limit)

    def patch_rerank(query, candidates, profile_summary, top_k=10, **kwargs):
        return candidates[:top_k]
    run_ablation("No Cross-Encoder Re-ranking (Task B)", "agents.rerank_agent.cross_encoder_rerank", patch_rerank, "b", args.limit)
