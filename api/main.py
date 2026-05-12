import asyncio

from fastapi import FastAPI, HTTPException
from api.schemas import (
    SimulateReviewRequest,
    SimulateReviewResponse,
    RecommendRequest,
    RecommendResponse,
)
from graphs.task_a import user_modeling_agent   # correct export name
from graphs.task_b import task_b_graph

app = FastAPI(title="DSN x BCT LLM Agent API")


@app.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(request: SimulateReviewRequest):
    """
    Run the user modelling pipeline (Task A) to predict a rating
    and generate a culturally-grounded simulated review.
    """
    initial_state = {
        # Align with UserAgentState schema in graphs/task_a.py
        "user_persona": request.user_id,
        "item_metadata": request.item.model_dump(),
    }

    try:
        # Run the synchronous LangGraph pipeline off the event loop
        # to avoid blocking FastAPI's async worker.
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
        "n": request.n,
    }

    try:
        result = await asyncio.to_thread(task_b_graph.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecommendResponse(recommendations=result.get("ranked_recommendations", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
