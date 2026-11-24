---
name: pytest-gen
description: Generates PEP 8 compliant pytest unit test suites with mocked Ollama, ChromaDB, and subprocess dependencies, then executes pytest locally to verify all tests pass.
---

# Pytest Unit Test Generation & Verification Workflow

When asked to write, update, or generate unit tests for any Python module in `/home/wrichter/Documents/Code/Projects/Python/processors/`, follow these steps sequentially:

## Step 1: Analyze Target Module & Mirror Structure
1. Inspect the target module using your PSI/file tools to identify all classes, `__init__` attributes, public methods, standalone functions, and edge-case exception paths.
2. Place test files inside `/home/wrichter/Documents/Code/Projects/Python/processors/tests/` mirroring the source name (e.g., `AI/mcp_workstation_server.py` -> `tests/test_mcp_workstation_server.py`). Ensure `tests/__init__.py` exists.

## Step 2: Enforce Offline Mocking & Design Standards
1. **Zero Live Hardware/Network Calls:** Never trigger real Ollama HTTP requests (`127.0.0.1:11434` or `11435`), live `nvidia-smi` calls, or persistent ChromaDB disk writes during unit tests. Use `pytest`'s `monkeypatch`, `tmp_path`, and `unittest.mock.patch` / `MagicMock`.
2. **PEP 8 & Documentation:** Follow all repository `AGENTS.md` standards—`test_<function>_<scenario>` naming, type hints on fixtures/helpers, and concise docstrings explaining what each test verifies.
3. **Coverage Checklist:**
   - Default and custom `__init__` state verification (confirming predefined attributes are initialized properly).
   - Happy-path execution with valid inputs.
   - Edge cases (empty inputs, missing files, invalid paths outside `repo_root`).
   - Error handling (network timeouts, non-zero subprocess exit codes, malformed payloads).

## Step 3: Execute & Verify via Terminal
1. Run `pytest` using the project's native Ubuntu virtual environment:
   ```bash
   /home/wrichter/Documents/Code/Projects/Python/CPU/.venv/bin/pytest tests/ -v
   ```
2. If `pytest` is not yet installed in that virtual environment, install it via `/home/wrichter/Documents/Code/Projects/Python/CPU/.venv/bin/pip install pytest` and re-run.
3. Fix any failing assertions, import path issues, or lint warnings until the suite exits with `0`.

