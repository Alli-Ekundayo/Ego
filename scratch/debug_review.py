import json
from pathlib import Path
from graphs.task_a import user_modeling_agent

def debug_review():
    with open("data/user_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    
    # Take the first test case
    profile = profiles[0]
    review = profile["test_reviews"][0]
    
    item_metadata = {
        "name": review.get("product_name", "Unknown"),
        "category": review.get("category", "Unknown"),
        "description": "",
    }
    
    result = user_modeling_agent.invoke({
        "user_persona": profile["name"],
        "item_metadata": item_metadata,
    })
    
    print("\n--- GROUND TRUTH ---")
    print(f"Rating: {review.get('rating')}")
    print(f"Review: {review.get('title')} {review.get('body')}")
    
    print("\n--- PREDICTED ---")
    print(f"Rating: {result['predicted_rating']}")
    print(f"Review: {result['final_review']}")

if __name__ == "__main__":
    debug_review()
