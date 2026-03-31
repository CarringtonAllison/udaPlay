# UdaPlay — AI Video Game Research Agent

UdaPlay is an AI-powered research assistant that answers natural language questions about video games. It uses a two-tier retrieval system: a local **ChromaDB vector database** for fast semantic search, falling back to **Tavily web search** when internal knowledge is insufficient. Web results are automatically persisted back to the knowledge base for long-term memory growth.

---

## Features

- **Dynamic data collection** — fetch structured game metadata for any title using Tavily + Claude (no hardcoded data)
- **RAG pipeline** — semantic search over a ChromaDB vector store with `sentence-transformers` embeddings
- **Confidence evaluation** — Claude scores retrieval quality and decides when a web search is needed
- **Web fallback + memory** — Tavily search results are normalized and persisted to ChromaDB for future queries
- **State machine agent** — clean `IDLE → RETRIEVE → EVALUATE → [ANSWER | WEBSEARCH → PERSIST] → ANSWER` workflow
- **Multi-turn sessions** — conversation history and Claude-generated summaries across turns
- **Structured output** — answers in natural language, JSON, or both

---

## Project Structure

```
udaPlay/
├── data/
│   └── games/
│       └── games.json              # Populated by Notebook 00
├── src/
│   ├── embeddings.py               # EmbeddingManager (sentence-transformers / OpenAI)
│   ├── vector_store.py             # VectorStoreManager + RetrievalResult
│   ├── data_collector.py           # GameDataCollector — Tavily search + Claude extraction
│   ├── tools.py                    # retrieve_game, evaluate_retrieval, game_web_search
│   ├── agent.py                    # UdaPlayAgent state machine + AgentContext
│   ├── memory.py                   # AgentMemory — session history + summaries
│   └── reporter.py                 # ReportFormatter — text / JSON / both
├── notebooks/
│   ├── Udaplay_00_data_collection.ipynb   # Collect game data from any seed list
│   ├── Udaplay_01_solution_project.ipynb  # Part 1: RAG pipeline
│   └── Udaplay_02_solution_project.ipynb  # Part 2: Agent implementation
├── tests/                          # 76 pytest unit tests (all mocked, no real API calls)
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

### 3. Collect game data (Notebook 00)

Open `notebooks/Udaplay_00_data_collection.ipynb` and run all cells.

Edit the `SEED_GAMES` list to collect data for any games you want:

```python
SEED_GAMES = [
    "Elden Ring",
    "Hollow Knight",
    "Stardew Valley",
    # ... add any game title here
]
```

The collector searches the web via Tavily, then asks Claude to extract and normalize the metadata into a structured JSON schema. Results are saved to `data/games/games.json`.

### 4. Build the RAG index (Notebook 01)

Open `notebooks/Udaplay_01_solution_project.ipynb` and run all cells.

This embeds the collected game data and loads it into a persistent ChromaDB vector database at `data/chroma_db/`.

### 5. Run the agent (Notebook 02)

Open `notebooks/Udaplay_02_solution_project.ipynb` and run all cells to see:

- Tool demos for `retrieve_game`, `evaluate_retrieval`, and `game_web_search`
- Full agent runs on example queries
- Structured output in text and JSON
- Multi-turn conversation sessions

---

## Agent Workflow

```
User Query
    │
    ▼
RETRIEVE ──► ChromaDB semantic search
    │
    ▼
EVALUATE ──► Claude scores confidence (0.0 – 1.0)
    │
    ├── confidence ≥ 0.65 ──► ANSWER
    │
    └── confidence < 0.65 ──► WEBSEARCH (Tavily)
                                  │
                              PERSIST (web results → ChromaDB)
                                  │
                               ANSWER
```

The confidence threshold is configurable via the `CONFIDENCE_THRESHOLD` environment variable (default: `0.65`).

---

## Running Tests

```bash
pytest tests/ -v
```

All 76 tests run without real API calls — Anthropic and Tavily clients are fully mocked. The `sentence-transformers` model is loaded once per session from the local cache.

```
76 passed in ~15s
```

---

## Game Data Schema

Each game document stored in `games.json` and ChromaDB:

| Field | Type | Description |
|---|---|---|
| `game_id` | `str` | URL-safe slug (e.g. `"elden-ring"`) |
| `title` | `str` | Full game title |
| `developer` | `str` | Development studio |
| `publisher` | `str` | Publishing company |
| `release_date` | `str` | ISO date (`YYYY-MM-DD`) or `"Unknown"` |
| `platforms` | `list[str]` | Platforms the game is available on |
| `genre` | `list[str]` | Genre tags |
| `description` | `str` | 2–4 sentence summary (embedded for RAG) |
| `metacritic_score` | `int` | Score 0–100 |
| `esrb_rating` | `str` | Age rating |
| `notable_features` | `list[str]` | Key selling points |
| `source` | `str` | `"internal"` or `"web_search"` |

---

## Example Queries

```
"Who developed FIFA 21?"
"When was God of War Ragnarök released?"
"What platforms is Hollow Knight available on?"
"What is Rockstar Games working on right now?"
"What are the most anticipated games coming out in 2025?"
```

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `TAVILY_API_KEY` | — | Required |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `sentence-transformers` or `openai` |
| `OPENAI_API_KEY` | — | Only needed if using OpenAI embeddings |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | Where ChromaDB stores its files |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this → web search fallback |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use |

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | [Anthropic Claude](https://www.anthropic.com) (`claude-sonnet-4-6`) |
| Embeddings | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| Vector database | [ChromaDB](https://www.trychroma.com) |
| Web search | [Tavily](https://tavily.com) |
| Data processing | pandas |
| Testing | pytest |