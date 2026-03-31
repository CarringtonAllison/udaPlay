# UdaPlay — AI Video Game Research Agent

UdaPlay is an AI-powered research assistant that answers natural language questions about video games. It uses a two-tier retrieval system: a local **ChromaDB vector database** for semantic search, falling back to **Tavily web search** when the knowledge base has insufficient information.

---

## Project Structure

```
udaPlay/
├── starter/
│   ├── games/
│   │   ├── 001.json          # Gran Turismo
│   │   ├── 002.json          # Grand Theft Auto: San Andreas
│   │   ├── ...
│   │   └── 015.json          # Halo Infinite
│   ├── lib/
│   │   ├── agents.py         # Agent class + AgentState TypedDict
│   │   ├── documents.py      # Document + Corpus
│   │   ├── evaluation.py     # AgentEvaluator + evaluation metrics
│   │   ├── llm.py            # LLM wrapper (OpenAI gpt-4o-mini)
│   │   ├── loaders.py        # PDFLoader
│   │   ├── memory.py         # ShortTermMemory + LongTermMemory
│   │   ├── messages.py       # Message types (System, User, AI, Tool)
│   │   ├── parsers.py        # Output parsers (Str, JSON, Pydantic)
│   │   ├── rag.py            # RAG pipeline + RAGState
│   │   ├── state_machine.py  # StateMachine, Step, Run, Snapshot, Resource
│   │   ├── tooling.py        # Tool class + @tool decorator
│   │   └── vector_db.py      # VectorStore + VectorStoreManager (ChromaDB)
│   ├── Udaplay_01_solution_project.ipynb   # Part 1: RAG pipeline
│   └── Udaplay_02_solution_project.ipynb   # Part 2: Agent implementation
├── tests/                    # 81 pytest tests (all mocked, no real API calls)
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
OPENAI_API_KEY="YOUR_KEY"
CHROMA_OPENAI_API_KEY="YOUR_KEY"
TAVILY_API_KEY="YOUR_KEY"
```

### 3. Run Notebook 01 — RAG Pipeline

Open `starter/Udaplay_01_solution_project.ipynb` and run all cells.

Loads the 15 game JSON files from `starter/games/`, embeds them with OpenAI embeddings, and stores them in a ChromaDB vector store. Demonstrates semantic search and the RAG pipeline.

### 4. Run Notebook 02 — Agent

Open `starter/Udaplay_02_solution_project.ipynb` and run all cells.

Demonstrates the full AI research agent with tool use, multi-turn sessions, and evaluation.

---

## Agent Workflow

```
User Query
    │
    ▼
Agent (gpt-4o-mini)
    │
    ├── retrieve_game() ──► ChromaDB semantic search
    │       │
    │       └── result returned to agent
    │
    └── game_web_search() ──► Tavily web search (fallback)
            │
            └── result returned to agent
                    │
                    ▼
                Final Answer
```

State machine: `entry → message_prep → llm_processor → [tool_executor ↔ llm_processor] → termination`

---

## Game Data Schema

Each game is stored in its own JSON file (`starter/games/001.json` – `015.json`):

| Field | Type | Description |
|---|---|---|
| `Name` | `str` | Full game title |
| `Platform` | `str` | Platform the game runs on |
| `Genre` | `str` | Genre classification |
| `Publisher` | `str` | Publishing company |
| `Description` | `str` | 2-3 sentence summary |
| `YearOfRelease` | `int` | Release year |

---

## Running Tests

```bash
pytest tests/ -v
```

All 81 tests run without real API calls — OpenAI and Tavily clients are fully mocked.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | OpenAI (`gpt-4o-mini`) |
| Embeddings | OpenAI `text-embedding-3-small` via ChromaDB |
| Vector database | ChromaDB (in-memory) |
| Web search | Tavily |
| State machine | Custom (`lib/state_machine.py`) |
| Testing | pytest |
