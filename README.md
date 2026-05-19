# Ego: Agentic Framework for Personalized Product Review Generation & Recommendations

A sophisticated LLM-powered agentic system that generates culturally-grounded product reviews and delivers personalized recommendations. Built with **LangGraph**, **LangChain**, and **Qdrant**, Ego combines user profiling, semantic similarity, and multi-agent workflows to produce authentic product feedback.

## 🎯 Overview

Ego solves two key challenges in e-commerce:

1. **Task A: Personalized Review Generation**
   - Predict ratings based on user history and item context
   - Generate reviews in the user's authentic voice
   - Inject culturally-grounded language (Nigerian Pidgin for Naija context)

2. **Task B: Smart Recommendations**
   - Retrieve candidate items using semantic search
   - Re-rank with LLM-powered reasoning
   - Handle cold-start users with synthetic profiles

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Endpoint                         │
│  POST /simulate-review    |    POST /recommend              │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
    ┌────────▼──────────┐        ┌─────────▼──────────┐
    │    Task A Graph   │        │   Task B Graph     │
    │  (Review Gen)     │        │ (Recommendations)  │
    └────────┬──────────┘        └─────────┬──────────┘
             │                              │
    ┌────────▼──────────┐        ┌─────────▼──────────┐
    │  5 Workflow Nodes │        │  5 Workflow Nodes  │
    │  • Profile        │        │  • Load Profile    │
    │  • Rating         │        │  • Cold-Start      │
    │  • Style          │        │  • Context Extract │
    │  • Generation     │        │  • Retrieve        │
    └────────┬──────────┘        │  • Rerank          │
             │                   └─────────┬──────────┘
             │                             │
    ┌────────▼─────────────────────────────▼───────────┐
    │         Shared Core Infrastructure               │
    │                                                  │
    │  • LLM Layer: Google Generative AI (cached)     │
    │  • Embeddings: Sentence Transformers            │
    │  • Vector Store: Qdrant (user profiles, items)  │
    │  • Config & Validation: Pydantic                │
    └──────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Ego/
├── api/                      # FastAPI application
│   ├── main.py              # Route handlers
│   └── schemas.py           # Request/response models
│
├── graphs/                   # LangGraph workflow definitions
│   ├── task_a.py            # Review generation pipeline
│   └── task_b.py            # Recommendation pipeline
│
├── agents/                   # Individual agent implementations
│   ├── naija_agent.py       # Nigerian cultural voice injection
│   ├── rating_agent.py      # Rating prediction algorithm
│   ├── retrieval_agent.py   # Candidate retrieval
│   ├── rerank_agent.py      # LLM-based re-ranking
│   ├── style_agent.py       # User writing style analysis
│   └── __init__.py          # Agent module exports
│
├── core/                     # Shared infrastructure
│   ├── config.py            # Settings & validation
│   ├── llm.py               # Cached LLM instance
│   ├── embeddings.py        # Embedding model wrapper
│   ├── vector_store.py      # Qdrant client wrapper
│   ├── user_profile.py      # Profile data structures
│   ├── math_utils.py        # Cosine similarity, etc.
│   └── __init__.py
│
├── data/                     # Static data & profiles
│   └── user_profiles.json   # Pre-built user profiles
│
├── tests/                    # Test suite
│   └── test_api.py
│
├── scripts/                  # Utility scripts
│   ├── build_user_profiles.py
│   ├── seed_naija_examples.py
│   └── evaluate.py
│
├── requirements.txt          # Python dependencies
├── conftest.py              # Pytest configuration
├── docker-compose.yml       # Qdrant + app services
└── Dockerfile               # Container image
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for Qdrant)
- Google API key (for Google Generative AI)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Alli-Ekundayo/Ego.git
   cd Ego
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env  # (or create manually)
   # Edit .env with your Google API key and other settings
   ```

5. **Start Qdrant vector store**
   ```bash
   docker-compose up -d qdrant
   ```

6. **Run the API server**
   ```bash
   python -m uvicorn api.main:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`

### First Test

**Generate a review** (Task A):
```bash
curl -X POST "http://localhost:8000/simulate-review" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "john_doe",
    "item": {
      "name": "Samsung 65-inch TV",
      "category": "Electronics",
      "description": "4K Smart TV with HDR support"
    }
  }'
```

Expected response:
```json
{
  "rating": 4.2,
  "review": "The picture quality is impressive...",
  "naija_review": "Abeg, this TV is serious! E don blow my mind with the clarity..."
}
```

**Get recommendations** (Task B):
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "john_doe",
    "context": "I need good earbuds for gaming",
    "n": 5
  }'
```

## 🔧 Core Components

### 1. Task A: Review Generation Pipeline

Located in `graphs/task_a.py` — a 4-node LangGraph workflow:

**Node Sequence:**
```
1. profile_retrieval_node
   └─ Load user profile & embeddings from Qdrant
   └─ Rank historical reviews by semantic similarity to target item
   └─ Fetch Nigerian voice examples for style injection

2. rating_prediction_node
   └─ Compute similarity-weighted average of past ratings
   └─ Blend with user's historical mean (70/30 split)
   └─ Clamp to [1.0, 5.0] range

3. style_analysis_node
   └─ Analyze user's writing patterns (sentence length, tone, phrases)
   └─ Generate a 2-3 sentence style profile
   └─ Falls back to statistical analysis if LLM unavailable

4. review_generation_node
   └─ Use few-shot examples to generate review in user's voice
   └─ Ensure review sentiment matches predicted rating
   └─ Include retry logic (3 attempts with exponential backoff)
```

**Key Features:**
- **Similarity-weighted rating**: Uses semantic embeddings to weight similar past reviews
- **Style consistency**: Analyzes and reproduces user's writing patterns
- **Fallback resilience**: Graceful degradation if services fail
- **Character truncation**: Keeps prompts within token budget

### 2. Task B: Recommendation Pipeline

Located in `graphs/task_b.py` — a 5-node LangGraph workflow with conditional routing plus multi-turn refinement:

**Node Sequence:**
```
1. context_extraction_node
   └─ Parse persona + conversation history into explicit and implicit preference signals
   └─ Extract current context fields (mood, occasion, location)
   └─ Infer domain and respect explicit domain filters

2. candidate_retrieval_node
   └─ Dual retrieval:
      ├─ Semantic similarity via Qdrant user/item embedding search
      └─ Collaborative filtering from similar users' liked items
   └─ Merge and deduplicate into top-20 candidates

3. cold_start_node (conditional)
   └─ Map persona demographics/preferences to nearest user cluster
   └─ Use cluster centroid as proxy embedding for new users
   └─ Project cross-domain signals through a learned linear map

4. reranking_node
   └─ Prompt LLM with profile + context + candidates
   └─ Return personalized top-10 with one-sentence reasoning each

5. multiturn_node
   └─ Handles conversational refinements ("actually ...", "more ...")
   └─ Re-retrieves and reranks with updated context using graph state memory
```

**Key Features:**
- **Cold-start handling**: Graceful fallback for new users
- **Cross-domain**: Can recommend across categories
- **Context-aware**: Parses natural language requests
- **Personalized reasoning**: LLM explains why each item is recommended

### 3. Agent Implementations

#### Rating Agent (`agents/rating_agent.py`)
Predicts user ratings using a hybrid algorithm:
- **Content-based**: Cosine similarity between user & item embeddings
- **Collaborative**: Weighted average of similar historical ratings
- **User-specific**: Adjusts for rating distribution (mean, std, skew)

#### Naija Agent (`agents/naija_agent.py`)
Injects culturally-grounded Nigerian voice:
- Retrieves authentic Naija-style examples from vector store
- Uses style guides to match tone and phrasing
- Preserves core sentiment and rating

#### Retrieval Agent (`agents/retrieval_agent.py`)
Dual-path candidate retrieval:
- Warm users: Semantic search over personal history
- Cold users: Query expansion + generic item search

#### Rerank Agent (`agents/rerank_agent.py`)
LLM-powered personalization:
- Scores candidates by relevance to user profile & context
- Generates natural language reasoning for each item
- Respects category filters and user preferences

#### Style Agent (`agents/style_agent.py`)
Analyzes writing patterns:
- Tokenizes reviews to detect repeated phrases
- Computes average sentence length
- Generates statistical or LLM-based style profiles

### 4. Core Infrastructure

#### `core/llm.py`
- **Cached LLM instance**: Uses `lru_cache` to reuse ChatGoogleGenerativeAI across all agents
- **Temperature control**: Supports different temperatures for different tasks
- **Fallback safety**: Gracefully handles missing API keys in testing

#### `core/embeddings.py`
- Wraps SentenceTransformers for consistent embedding generation
- Supports batch embedding for efficiency
- Maintains embedding dimension info for fallback logic

#### `core/vector_store.py`
- Qdrant client wrapper with convenient methods
- `search()`: Semantic search by vector
- `retrieve_by_id()`: Direct ID-based lookups
- `upsert()`: Batch insertion of documents

#### `core/config.py`
- Pydantic-based settings management
- Environment variable support with defaults
- Secret value masking for API keys

## 📊 Workflow Details

### How Review Generation Works (Task A)

1. **Input**: User persona + item metadata
2. **Step 1 - Profile Retrieval**:
   - Compute stable MD5 hash of user name
   - Load user profile + embeddings from Qdrant
   - Fetch top-10 similar historical reviews (by cosine similarity)
   - Retrieve Nigerian voice examples

3. **Step 2 - Rating Prediction**:
   - Weight each similar review by its similarity score
   - Compute weighted average rating
   - Blend with user's historical mean (70% local, 30% global)
   - Clamp result to [1.0, 5.0]

4. **Step 3 - Style Analysis**:
   - Tokenize user's past reviews
   - Compute average sentence length and bigram frequencies
   - Use LLM to generate style description (or fallback to statistical profile)

5. **Step 4 - Review Generation**:
   - Use few-shot examples (up to 5 similar reviews)
   - Prompt LLM to generate new review matching rating & style
   - Retry up to 3 times with 2s, 4s backoff
   - Clean output by removing markdown, truncating sections

6. **Output**: Predicted rating + simulated review (both English & Nigerian-flavored)

### How Recommendations Work (Task B)

1. **Input**: User ID + context text + optional filters
2. **Step 1 - Profile Loading**:
   - Query Qdrant for user profile
   - If not found, flag as cold-start

3. **Step 2 - Branching**:
   - Warm users: Proceed to context extraction
   - Cold users: Build synthetic profile from persona

4. **Step 3 - Context Extraction**:
   - Parse user's request with LLM
   - Extract primary domain (or use explicit filter)

5. **Step 4 - Candidate Retrieval**:
   - For warm users: Semantic search over personal history embeddings
   - For cold users: Persona-based item search
   - Return top-100 candidates

6. **Step 5 - Re-ranking**:
   - Score each candidate using LLM scorer
   - Include personalized reasoning
   - Return top-N sorted by relevance

7. **Output**: Ranked list of recommendations with explanations

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run a specific test:
```bash
pytest tests/test_api.py::test_simulate_review -v
```

Smoke test the Task A graph:
```bash
cd graphs
python task_a.py  # Runs the embedded __main__ test
```

## 🔐 Environment Variables

Create a `.env` file in the project root:
```env
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM Configuration
LLM_MODEL=gemma-4-26b-a4b-it
LLM_TEMPERATURE=0.7

# Application Settings
API_PORT=8000
DEBUG=false
```

## 📚 Key Concepts

### Embeddings
- **What**: Dense vector representations of text
- **Why**: Enable semantic similarity without keyword matching
- **How**: Using SentenceTransformers (MiniLM for speed, all-mpnet for quality)

### Vector Store (Qdrant)
- **What**: Vector database for similarity search
- **Why**: Fast retrieval of similar items & users
- **Stored**: User profiles, item metadata, Naija style examples

### LangGraph
- **What**: Framework for building stateful, multi-agent workflows
- **Why**: Handles complex branching, retries, and state management
- **Used**: Define Task A & Task B as DAGs with typed state

### Cosine Similarity
- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Range**: [-1, 1] where 1 = identical direction
- **Application**: Weight historical reviews, rank candidates

### Cold-Start Problem
- **Problem**: New users have no history
- **Solution**: Build synthetic profile from persona description
- **Fallback**: Use generic item search + LLM re-ranking

## 🛠️ Development

### Adding a New Agent

1. Create a new class in `agents/new_agent.py`:
   ```python
   from core.llm import get_llm
   from core.config import settings

   class NewAgent:
       def __init__(self):
           self.llm = get_llm(settings.LLM_MODEL)
       
       def process(self, input_data: dict) -> str:
           # Implementation here
           pass
   
   # Export singleton instance
   new_agent = NewAgent()
   ```

2. Use in your workflow nodes:
   ```python
   from agents.new_agent import new_agent
   
   def my_node(state: MyState) -> dict:
       result = new_agent.process(state["data"])
       return {"result": result}
   ```

### Adding a New Workflow Node

1. Define the node function (takes state, returns dict):
   ```python
   def new_node(state: MyState) -> dict:
       """Process state and return updates."""
       # Your logic here
       return {"key": "value"}
   ```

2. Register in the graph:
   ```python
   workflow.add_node("new_node", new_node)
   workflow.add_edge("previous_node", "new_node")
   ```

### Debugging Workflows

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Trace execution step-by-step:
```python
result = graph.invoke(initial_state)
# Check intermediate state in result
```

## 📈 Performance Considerations

- **Embedding Batch Size**: Default 10 for memory efficiency
- **Vector Search Limit**: Default 3-10 similar items to reduce noise
- **Review Character Limit**: 300 chars per review to stay within token budget
- **LLM Retry**: 3 attempts with exponential backoff (2s, 4s)
- **Cache Duration**: LLM instances cached for entire process lifetime

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear comments
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License — see LICENSE file for details.

## 🙋 Support

For questions or issues:
- Check existing GitHub issues
- Review inline code comments for implementation details
- Run tests to verify your setup

---

**Built with ❤️ by the DSN x BCT LLM Agent team**
