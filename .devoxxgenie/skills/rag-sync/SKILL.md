---
name: rag-sync
description: Automatically indexes and synchronizes newly created or modified Python, Markdown, and configuration files into the local CPU ChromaDB vector database via the workstation-ai MCP server.
---

# Local ChromaDB RAG Synchronization Workflow

Whenever you create, refactor, or update files in `/home/wrichter/Documents/Code/Projects/Python/processors/`, execute this synchronization workflow before finishing your task:

## Step 1: Identify Modified Files
1. Collect the relative file paths (relative to `/home/wrichter/Documents/Code/Projects/Python/processors/`) of every `.py`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`, or `.modelfile` file you created or edited during the current session.
2. Ignore temporary files, `__pycache__`, `.venv`, and ChromaDB storage folders.

## Step 2: Re-Index via MCP Tool
1. For each modified file, call the `ingest_code_to_rag` tool from the `workstation-ai` MCP server:
   - `target_path`: The relative path of the modified file (e.g., `"AI/multiAgent.py"` or `"main_processor.py"`).
   - `replace_existing`: `true` (ensures stale chunks for that file are purged before inserting updated embeddings).
2. If you refactored an entire directory, pass the folder path (e.g., `"AI/"`) instead of calling the tool one file at a time.

## Step 3: Verify & Report
1. Confirm the tool returns an `Ingestion complete:` status with non-zero chunk counts.
2. If the tool reports a connection error to `127.0.0.1:11435`, inform the user that the secondary CPU Ollama instance needs to be started (`CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:11435 ollama serve`).
3. Include a brief one-line confirmation in your final response listing which files were synced to `chromeLangChainCodeDb`.

