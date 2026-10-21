# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG chatbot that answers questions about course materials. It is a FastAPI backend with a vanilla JS frontend. ChromaDB stores the vectors, sentence-transformers (`all-MiniLM-L6-v2`) creates the embeddings, and Claude generates the answers through tool calling.

## Commands

- Install dependencies: `uv sync` (needs Python >= 3.13)
- Configure: create a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`)
- Run: `./run.sh`, or `cd backend && uv run uvicorn app:app --reload --port 8000`
  - Web UI: http://localhost:8000. API docs: http://localhost:8000/docs

- Tests (pytest is a dev dependency; config lives in `pyproject.toml`): run `uv run pytest` from the repo root. For a single test: `uv run pytest backend/tests/test_document_processor.py::TestChunkText::test_whitespace_is_normalized`
  - `backend/tests/test_document_processor.py` holds characterization tests. Tests named `pinned_quirk` lock in current behavior that is probably a bug. Update them on purpose when fixing that behavior.
  - Golden snapshots of the parsed `docs/*.txt` are stored in `backend/tests/snapshots/`. After an *intentional* change to parsing or chunking output, regenerate them with `UPDATE_SNAPSHOTS=1 uv run pytest`.

The repo has no linter or build step. The root `main.py` is an unused placeholder.

**Run the server from inside `backend/`.** The code uses paths relative to the working directory: `../docs` for documents, `../frontend` for static files, and `./chroma_db` for the vector DB (so the DB lives at `backend/chroma_db`).

## Architecture

The flow for a single request is `app.py` → `RAGSystem` (`rag_system.py`) → `AIGenerator` → Claude → `ToolManager` → `CourseSearchTool` → `VectorStore`.

- **Retrieval is tool-based, not prompt-stuffed.** `RAGSystem.query` never runs a search itself. It passes the tool definitions to `AIGenerator.generate_response`, and Claude decides whether to call `search_course_content`. On `stop_reason == "tool_use"`, `_handle_tool_execution` runs the tools and makes one more API call with the same `tools` but `tool_choice: none`. That means there is at most one round of tool calls per query, which the system prompt reinforces ("One search per query maximum").
- **Sources are passed back through side-channel state.** `CourseSearchTool._format_results` stores `last_sources` on the tool instance. `RAGSystem.query` then reads them with `ToolManager.get_last_sources()` and clears them with `reset_sources()`. A new tool that should report sources must expose a `last_sources` attribute. Because this is shared mutable state on a singleton, concurrent requests can mix up sources.
- **There are two Chroma collections** in `vector_store.py`:
  - `course_catalog`: one document per course. The ID is the course title, and lessons are serialized as JSON in the `lessons_json` metadata field. It is used for *fuzzy course-name resolution*: `_resolve_course_name` runs a vector search over titles so that partial names like "MCP" work.
  - `course_content`: text chunks with `course_title`, `lesson_number` and `chunk_index` metadata. `search()` first resolves the course name, then filters content with a `where` clause.
- **Course title is the primary key everywhere.** Chunk IDs are `{title_with_underscores}_{chunk_index}`.
- **Ingestion happens on startup** (`app.py` `startup_event` calls `add_course_folder("../docs", clear_existing=False)`). It skips any course whose title already exists in Chroma, so edits to an existing doc are *not* re-indexed. To pick up changes, delete `backend/chroma_db` or call `add_course_folder(..., clear_existing=True)`.
- **Document format** (parsed by `document_processor.py`): the first lines are `Course Title:`, `Course Link:` and `Course Instructor:`. After those, each `Lesson N: <title>` marker can be followed by a `Lesson Link:` line, then the lesson text. Chunks are built from sentences (`CHUNK_SIZE` 800 chars, `CHUNK_OVERLAP` 100). Note: the context prefix added to chunks is inconsistent. For most lessons only the first chunk gets the `Lesson N content:` prefix. For the *last* lesson, every chunk gets `Course <title> Lesson N content:`. `.pdf` and `.docx` files are accepted by extension but are read as plain UTF-8 text.
- **Sessions** (`session_manager.py`) live in memory only and are lost on restart. History is capped at `MAX_HISTORY` exchanges and is injected into the *system prompt* as formatted text, not as message turns.
- **Configuration** is in `backend/config.py`: the model, chunking, `MAX_RESULTS`, `MAX_HISTORY` and the DB path. The model is `claude-sonnet-5-5`. Its generation settings are in `AIGenerator.base_params`.
- **Sonnet 5.5 constraints** (`ai_generator.py`):
  - It always runs adaptive thinking, so responses can start with `thinking` blocks. Read text through `_extract_text`, never `content[0]`.
  - Thinking counts toward `max_tokens` (16000). Depth is controlled by `effort` (`low`).
  - A non-default `temperature` returns a 400.
  - `tool_choice` `any`/`tool` returns a 400.
  - Changing `tools` or `system` between the tool call and the follow-up call invalidates the replayed thinking blocks.
  - The pinned SDK (`anthropic==0.58.2`) predates `output_config` and `fallbacks`, so they are sent via `extra_body`/`extra_headers`.
- **Frontend** (`frontend/script.js`) calls only `POST /api/query` (with `{query, session_id}`) and `GET /api/courses`. FastAPI serves it as static files mounted at `/`. A `DevStaticFiles` no-cache class is defined in `app.py` but is not used.
