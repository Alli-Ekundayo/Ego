import asyncio

from fastapi import FastAPI, HTTPException

from api.schemas import (
    RecommendRequest,
    RecommendResponse,
    SimulateReviewRequest,
    SimulateReviewResponse,
)
from graphs.task_a import user_modeling_agent
from graphs.task_b import task_b_graph

app = FastAPI(title="Ego User Modelling Agent API")


@app.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(request: SimulateReviewRequest):
    """
    Run the user modelling pipeline (Task A) to predict a rating
    and generate a culturally-grounded simulated review.
    """
    initial_state = {
        "user_persona": request.user_id,
        "item_metadata": request.item.model_dump(),
    }

    try:
        result = await asyncio.to_thread(user_modeling_agent.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SimulateReviewResponse(
        rating=result["predicted_rating"],
        review=result.get("simulated_review", ""),
        naija_review=result.get("final_review"),
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Run the retrieval + rerank pipeline (Task B) to return N recommendations.
    """
    initial_state = {
        "user_id": request.user_id,
        "context_text": request.context,
        "persona_description": request.persona_description,
        "session_history": request.session_history,
        "domain_filter": request.domain_filter,
        "n": request.n,
    }

    try:
        result = await asyncio.to_thread(task_b_graph.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    recommendations = (
        result.get("ranked_recommendations") or result.get("recommendations") or []
    )

    # Flatten if it's a list of lists (sometimes happens with certain LLM outputs)
    if recommendations and isinstance(recommendations[0], list):
        recommendations = [item for sublist in recommendations for item in sublist]

    normalised = []
    for rec in recommendations:
        if isinstance(rec, dict):
            normalised.append(
                {
                    "item_id": str(rec.get("item_id", "")),
                    "name": str(rec.get("name", "Unknown")),
                    "reason": str(rec.get("reason", rec.get("reasoning", ""))),
                }
            )

    return RecommendResponse(recommendations=normalised)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
