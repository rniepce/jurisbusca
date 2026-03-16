# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**JurisBusca** is a Brazilian legal AI assistant for judicial document processing. It ingests court case PDFs, indexes them semantically (ChromaDB), and uses multiple LLM pipelines to draft legal decisions, rulings, and petitions.

## Commands

### Backend (Python / FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (primary entrypoint)
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# Or use the start script (auto-activates venv312 with MLX on Apple Silicon)
./start.sh

# Legacy Streamlit interface (deprecated, kept for reference)
python -m streamlit run app.py
```

### Frontend (React / Vite)

```bash
cd frontend

# Install
npm ci

# Development server (proxies /api to localhost:8000)
npm run dev

# Build for production (output goes to frontend/dist/, served by FastAPI)
npm run build

# Lint
npm run lint
```

### Docker

```bash
docker build \
  --build-arg VITE_SUPABASE_URL=<url> \
  --build-arg VITE_SUPABASE_ANON_KEY=<key> \
  -t jurisbusca .

docker run -p 8000:8000 --env-file .env jurisbusca
```

## Architecture

### Request Flow

```
Browser → FastAPI (api_server.py)
            ├── Serves React frontend (frontend/dist/) as static files
            ├── /api/* → orchestration dispatched through backend.py
            └── Auth: Supabase JWT validated on every protected route
```

### Python Layer

| File | Role |
|---|---|
| `api_server.py` | FastAPI app, all HTTP routes, SSE streaming, remote SLM proxy |
| `backend.py` | Core orchestration — LLM wiring, document ingestion, ChromaDB, AI engine dispatch |
| `history_db.py` | SQLite (thread-local connections): conversations, messages, user memories |
| `ocr_engine.py` | PaddleOCR + PyMuPDF for scanned PDFs |
| `chunking.py` | Hybrid semantic chunker for document splitting |
| `raptor_engine.py` | Hierarchical RAPTOR indexing over ChromaDB |
| `planning_engine.py` | Pre-processing planner for complex queries |
| `style_engine.py` | Analyzes judge's writing style from template documents |
| `slm_engine.py` | Apple Silicon MLX inference (local SLMs) |
| `slm_orchestrator.py` | Multi-step pipeline using local SLMs |
| `slm_server.py` | FastAPI server exposing local MLX models (run on MacBook, tunneled to Railway) |

### AI Engines (selectable per request)

- **V1 (Gemini)** — Single-LLM, ChromaDB RAG. Runs entirely in `backend.py`.
- **V2 (Multi-Agent, LangGraph)** — `v2_engine/orchestrator_v2.py`. Three sequential Claude agents: Triage → Drafting → Revision.
- **V3 (Autonomous Magistrate, LangGraph MoE)** — `v3_engine/orchestrator_v3.py`. Mixture of Experts: Kimi K2.5 (facts extraction) → DeepSeek (drafting with tool loop via `LegalREPL`) → Claude/GPT (QA).
- **SLM Local** — MLX models on Apple Silicon via `slm_engine.py`. Disabled on Railway.
- **Remote SLM** — `SLM_SERVER_URL` points Railway to a local MacBook SLM server over a tunnel.

### Frontend (React + Vite)

- `frontend/src/services/api.js` — all backend calls; attaches Supabase JWT and Azure key headers.
- `frontend/src/services/supabase.js` — Supabase client.
- `frontend/src/components/AuthContext.jsx` — auth context provider.
- Components in `frontend/src/components/` are standalone UI panels: Chat, Batch, XRay, Jurisprudence, Memory, ModelManager, CanvasEditor.

### Prompts

All LLM prompts live in dedicated root-level files:
- `prompts.py` — V1/V2 base prompts
- `prompts_claude.py` — Claude-specific prompts (integral analysis, auditor, X-Ray, style)
- `prompts_magistrate_v3.py` — V3 magistrate core prompt
- `prompts_auditor.py`, `prompts_slm.py` — auditor/SLM-specific prompts

### Persistence

- **ChromaDB** — vector store for semantic search; path from `CHROMA_DB_PATH` env var (default `./chroma_db/`, mapped to Railway Volume at `/chroma_data`). Gitignored — must be regenerated or uploaded per deployment.
- **SQLite** — `history.db` for conversation history and user memories; path from `HISTORY_DB_PATH`.

## Environment Variables

```
# LLM Providers
GOOGLE_API_KEY          # Gemini (V1 default)
OPENAI_API_KEY          # OpenAI direct
AZURE_OPENAI_API_KEY    # Azure OpenAI (V2/V3)
ANTHROPIC_API_KEY       # Claude (V2/V3)
DEEPSEEK_API_KEY        # V3 drafting expert

# Auth
SUPABASE_URL
SUPABASE_ANON_KEY
SUPABASE_SERVICE_KEY    # Backend JWT validation

# Storage
CHROMA_DB_PATH          # ChromaDB directory (Railway: /chroma_data)
HISTORY_DB_PATH         # SQLite path (default: history.db)
JURISPRUDENCIA_DB_PATH  # Pre-built jurisprudence vector DB

# SLM Tunnel (optional)
SLM_SERVER_URL          # URL of local slm_server.py exposed via tunnel
SLM_SERVER_KEY          # Shared secret for tunnel auth

PORT                    # Server port (default: 8000)
```

## Deployment

Deploys to **Railway** via Docker (`Dockerfile`) or Nixpacks (`nixpacks.toml`). The `Procfile` is a legacy Railway fallback. Railway Volumes must be mounted at `/chroma_data` for vector DB persistence.

Local SLMs (MLX) require `pip install mlx mlx-lm` in `venv312` (Apple Silicon only). Run `slm_server.py` separately and expose via tunnel (ngrok/Cloudflare), then set `SLM_SERVER_URL`.
