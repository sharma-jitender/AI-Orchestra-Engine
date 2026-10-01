# LLM Orchestration Engine

Middleware between an app and multiple models. A FastAPI gateway classifies each prompt, scores models on quality/cost/latency, calls a uniform adapter, falls back on failure, and logs everything to PostgreSQL. A React dashboard shows routing mix and a **baseline comparison** (always-strong model vs routed policy).

This is Solution 1 from the architecture discussion: a scoring monolith plus an eval harness. No ML classifier, no bandit, no per-request judge model.

## Quick start

Local default is **SQLite** so the demo runs without Docker. `docker-compose.yml` still ships **PostgreSQL** for the full stack. Set `DATABASE_URL` to the Postgres URL when Compose is available.

```bash
docker compose up postgres -d
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
uvicorn app.main:app --reload --app-dir .
```

If you run uvicorn from `backend/`, use:

```bash
cd backend
uvicorn app.main:app --reload
```

**Dashboard:**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Click **Run baseline eval** — that is the proof. No API keys required; mock adapters are used until `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are set. Ollama is optional (`OLLAMA_BASE_URL`).

Full stack: `docker compose up --build`.

## What is in v1

| Piece | Behavior |
| --- | --- |
| Classifier | Rules + optional `task_type` hint (`qa`, `code`, `summarization`, `reasoning`, `creative`) |
| Registry | `backend/config/models.yaml` |
| Router | `score = w_q * strength - w_c * cost_norm - w_l * latency_norm` (`routing.yaml`) |
| Adapters | Mock, OpenAI, Anthropic, Ollama |
| Fallback | Ranked list, then mock if `ALLOW_MOCK_FALLBACK=true` |
| Logs | `request_logs` in Postgres |
| Eval | `POST /v1/eval/run` replays `backend/fixtures/eval_prompts.json` under both policies |

## API

- `POST /v1/complete` `{ "prompt", "task_type?", "policy": "routed"|"baseline", "force_model?" }`
- `GET /v1/stats/overview`
- `GET /v1/stats/by-model` / `by-task`
- `GET /v1/requests`
- `POST /v1/eval/run`
- `GET /v1/models` / `GET /v1/rank?task_type=code`

## Tests

```bash
cd backend
python -m pytest tests -q
```

(`pytest` is optional: `pip install pytest`.)
