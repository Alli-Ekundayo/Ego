import json
import hashlib
from core.vector_store import vector_store
from core.embeddings import embedding_model

class RetrievalAgent:
    def __init__(self):
        # Cache profiles in memory for fast CF lookup
        self.profiles_db = {}
        try:
            with open("data/user_profiles.json", "r", encoding="utf-8") as f:
                profiles = json.load(f)
                for p in profiles:
                    self.profiles_db[p["user_id"]] = p
        except Exception as e:
            print(f"Warning: Could not load user_profiles.json: {e}")

    def retrieve(self, context: str, user_id: str = None, n: int = 10) -> list[dict]:
        """
        Retrieves candidate items using Semantic + CF signals.
        - Semantically searches for users matching the context.
        - CF: Recommends the top-rated items from those similar users.
        """
        query_vector = embedding_model.embed_text(context)
        
        # Search Qdrant for similar users based on context
        try:
            results = vector_store.client.search(
                collection_name="user_profiles",
                query_vector=query_vector,
                limit=5
            )
        except Exception:
            return []
            
        candidates = []
        seen_product_ids = set()
        
        for res in results:
            sim_user_id = res.payload.get("id")
            if not sim_user_id or sim_user_id == user_id:
                continue
                
            sim_user = self.profiles_db.get(sim_user_id)
            if not sim_user:
                continue
                
            # CF Signal: Get their highly rated items
            for review in sim_user.get("train_reviews", []):
                prod_id = review.get("product_id")
                rating = review.get("rating", 0)
                
                if prod_id and prod_id not in seen_product_ids and rating >= 4.0:
                    candidates.append({
                        "item_id": prod_id,
                        "name": review.get("product_name"),
                        "category": review.get("category"),
                        "url": review.get("product_url"),
                        "source_user": sim_user.get("name"),
                        "cf_score": res.score * rating  # Combine semantic user match with their rating
                    })
                    seen_product_ids.add(prod_id)
                    
        # Sort by CF score and limit to n
        candidates.sort(key=lambda x: x["cf_score"], reverse=True)
        return candidates[:n]

retrieval_agent = RetrievalAgent()


def retrieve_candidates(profile, context_text: str, domain_filter: str | None = None) -> list[dict]:
    """
    Adapter used by Task B graph.
    The current retrieval backend is context-driven and does not require `profile`.
    """
    candidates = retrieval_agent.retrieve(context=context_text, user_id=getattr(profile, "user_id", None), n=20)
    if domain_filter:
        domain = domain_filter.strip().lower()
        candidates = [c for c in candidates if str(c.get("category", "")).strip().lower() == domain]
    return candidates


def cold_start_retrieval(persona_description: str, domain: str | None = None) -> list[dict]:
    """
    Cold-start retrieval uses the same semantic retrieval path with persona text as context.
    """
    candidates = retrieval_agent.retrieve(context=persona_description, user_id=None, n=20)
    if domain:
        dom = domain.strip().lower()
        candidates = [c for c in candidates if str(c.get("category", "")).strip().lower() == dom]
    return candidates
