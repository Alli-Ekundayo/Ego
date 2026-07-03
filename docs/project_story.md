# Ego: The MemoryAgent for Nigerian E-Commerce

A project story built for the **Global AI Hackathon with Qwen Cloud (Track 1: MemoryAgent)**.

---

## Inspiration

Modern recommendation engines are often frustratingly forgetful. They behave either as purely stateless machines—treating each new session as if they are meeting the user for the first time—or as rigid collaborative filters that reduce human nuance to a static feature vector. They ignore qualitative user expressions, immediate feedback loops, and cross-session context shifts. 

When looking at the **Nigerian e-commerce landscape**, this forgetfulness becomes even more costly. Shopping in Nigeria is highly dynamic, personal, and culturally nuanced. Users often communicate in a rich blend of Pidgin and local English phrases (which we call the "Naija voice"), and their purchasing decisions are heavily dictated by local infrastructural constraints:
* **Power availability**: Prioritizing massive power bank capacities or battery-efficient electronics.
* **Data conservation**: Looking for apps and gadgets that optimize internet bandwidth.
* **Economic sensitivity**: Relying on trusted value brands (e.g., Oraimo, Infinix, Tecno) while managing strict Naira budgets.
* **Trust & Logistics**: Searching for reliable sellers to avoid delivery challenges.

We were inspired to build **Ego**—meaning *money* in Igbo, symbolizing the commerce-centric nature of the project. We wanted to create a recommendation agent that acts like a highly attentive local shop assistant: one that remembers a user's verbal feedback, understands their colloquial preferences, dynamically adapts to their context, but also allows irrelevant details to fade naturally over time, mimicking human memory.

---

## What it does

Ego is a dual-task recommendation and user modeling system for Nigerian e-commerce, grounded in real product reviews scraped from Jumia Nigeria. It features a cognitive-inspired **MemoryAgent** that maintains a persistent, cross-session memory layer:

1. **MemoryAgent (Track 1 Focus)**: A persistent memory graph backed by SQLite. As the user interacts with Ego (searching, rating, reviewing), the system records interaction logs. In the background, it runs a **Qwen-Plus** powered consolidation pass to merge duplicates, extract structured key-value preferences, and update a rolling long-term semantic summary.
2. **Task A (User Modelling)**: Predicts ratings for unseen products and generates culturally grounded product reviews in an authentic "Naija voice" using statistical style profiling and historical reviews as exemplars.
3. **Task B (Recommendation)**: A multi-node LangGraph pipeline providing hybrid retrieval (Dense Semantic + Collaborative Filtering + BM25 keyword matching fused via Reciprocal Rank Fusion) and two-stage reranking (Cross-encoder sorting + LLM personalization).

---

## How we built it

Ego was built from the ground up using modular, production-ready AI tools:

```
                      ┌─────────────────────────────────────────────────────┐
                      │             FastAPI Gateway (Alibaba Cloud ECS)      │
                      │  /recommend  /simulate-review                        │
                      │  /memory/ingest  /memory/recall  /memory/snapshot   │
                      └──────┬─────────────────┬──────────────┬─────────────┘
                             │                 │              │ async fire-and-forget
                      Task B Graph      Task A Graph    MemoryAgent Graph
                      (Recommend)      (User Model)    (Persistent Memory)
                             │                 │              │
                ┌────────────┘      ┌──────────┘    ┌─────────┘
                ▼                   ▼               ▼
           Turbovec           Turbovec +       Memory SQLite DB
           Vector Store       Naija Examples   (cross-session)
                │                   │               │
                └────────────┬──────┘      Qwen Cloud (DashScope)
                             │             consolidation LLM
                       BM25 Corpus
```

### 1. Mathematical Forgetting Curve
Instead of relying on crude Time-To-Live (TTL) timestamps that expire memories abruptly, we implemented a cognitive model inspired by Hermann Ebbinghaus's spacing effect. The retention probability $R(t)$ of a memory decays exponentially over time $t$ (measured in days), moderated by how often it is recalled:

$$R(t) = \text{importance}_0 \times \exp\left(-\frac{t}{\text{half\_life}} \cdot \frac{1}{\sqrt{1 + \text{access\_count}}}\right)$$

* **Decay Rate**: We set $\text{half\_life} = 7.0$ days. A preference that is never re-accessed decays below the pruning threshold ($\theta = 0.05$) in approximately 37 days.
* **Access Boost**: Each time a memory is recalled during a recommendation pass, its metadata is updated:
  $$\text{access\_count} \leftarrow \text{access\_count} + 1$$
  $$\text{importance} \leftarrow \min(1.0, \text{importance} + 0.08)$$
  This resets the decay clock and increases its resistance to future forgetting, mirroring human cognitive consolidation.

### 2. Token-Budget-Aware Recall
To prevent LLM context-window bloating and control API costs, Ego uses a token-budget-aware recall mechanism. When querying memories, it scores each candidate by combining its decayed importance with its search relevance:

$$\text{Score} = R(t) \times \left(1.0 + 0.5 \times \text{keyword\_overlap}\right)$$

It then greedily fills the prompt space, packing the highest-scoring memories until it reaches the user-specified `max_tokens` limit.

### 3. Agent Orchestration & LLMs
* **LangGraph**: Orchestrates the state transitions of the MemoryAgent (`ingest_node` $\rightarrow$ `consolidate_node` $\rightarrow$ `prune_node`) and the search pipeline.
* **Qwen-Plus (Alibaba Cloud DashScope)**: Functions as our primary LLM for structured JSON extraction and long-term consolidation.
* **Google Gemini**: Acts as a transparent fallback in case of rate limits.
* **Turbovec**: A lightweight local vector store for dense retrieval, utilizing `all-MiniLM-L6-v2` embeddings.

---

## Challenges we ran into

### 1. Asynchronous Ingestion & Latency
Running LLM-based memory consolidation inside the synchronous request loop of a recommendation engine added unacceptable latency (often $>2$ seconds).
> **Solution**: We decoupled the MemoryAgent. In FastAPI, the `/recommend` route retrieves recommendations immediately and fires the memory ingestion graph in the background using `asyncio.create_task()`. The user gets sub-second response times, and the memory consolidates silently in the background.

### 2. JSON Contamination & Schema Enforcement
LLMs occasionally wrap JSON outputs in Markdown code blocks (e.g., ` ```json ... ``` `), which breaks standard Python `json.loads()`.
> **Solution**: We built a robust parser utilizing regex stripping:
> ```python
> stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
> stripped = re.sub(r"\s*```$", "", stripped)
> ```
> We also implemented database fallbacks to retrieve the last successfully consolidated memory summary if a new consolidation pass fails.

### 3. Multi-Turn Price Refinement Without Database Fields
During conversational multi-turn refinements, users might ask to see "cheaper alternatives" or "something more premium". However, raw review datasets lack clean price fields.
> **Solution**: We built regex parsers to extract numerical Naira prices from product strings (e.g., `₦ 25,000` $\rightarrow$ `25000.0`). We then compute the median price dynamically across all active candidates:
> $$\tilde{P} = \text{median}(P_{\text{candidates}})$$
> If the user asks for budget options, we filter the candidate set to only items where $P \le \tilde{P}$. If price data is entirely missing, we dynamically fallback to user rating averages as a proxy for product quality.

---

## Accomplishments that we're proud of

* **Sub-Second Recommendations**: Achieved highly personalized recommendations by shifting the heavy Qwen-powered consolidation graph into an asynchronous post-response hook.
* **Cultural Grounding ("Naija Voice")**: Task A successfully generates reviews containing natural Nigerian expressions (e.g., "Abeg", "Wahala", "No cap", "Oraimo is a lifesaver") without sounding forced. We achieved this by feeding similarity-ranked Jumia reviews directly into Qwen.
* **Mathematically Proven Forgetting**: A fully functioning SQLite memory store that decays, prunes, and caps user memories, keeping the context window clean.

---

## What we learned

1. **OpenAI Compatibility is a Superpower**: Alibaba Cloud DashScope’s OpenAI-compatible endpoint allowed us to drop Qwen-Plus into our existing codebase by changing only a couple of environment variables (`base_url` and `api_key`).
2. **Cognitive Tiering**: A good recommendation memory requires tiering. Raw logs (episodic memory) are too noisy for prompts. We need structured preferences (semantic memory) and rolling text summaries (narrative memory) working in tandem.
3. **Data-Centric Local Profiling**: To generate authentic local voices, stylistic profiles (e.g., calculating Type-Token Ratio for vocabulary richness and average sentence length) are faster and more reliable than prompting an LLM to guess a style.

---

## What's next for Ego

* **Collaborative Memory Clustering**: Grouping anonymous users with similar decayed memory profiles (e.g., "Lagos Tech Enthusiasts") to enable collaborative-style memory propagation.
* **Continuous Online Adaptation**: Fine-tuning local embedding models using a user's accumulated memory summaries to project search vectors closer to their preference space.
* **Pidgin Voice Interfaces**: Integrating local speech-to-text models so Nigerian shoppers can speak naturally to Ego in Pidgin, Yoruba, Igbo, or Hausa, with the MemoryAgent converting the voice input directly into structured preferences.
