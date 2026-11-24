"""Unit tests for AI/mcp_workstation_server.py covering RAG, path safety, and hardware tools."""

import io
import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import urllib.error

from langchain_core.documents import Document
import pytest

from AI.mcp_workstation_server import (
    WorkstationMCPBridge,
    get_gpu_vram_status,
    ingest_code_to_rag,
    search_project_rag,
    summarize_on_cpu_llm,
)


@pytest.fixture
def bridge(tmp_path: Path) -> WorkstationMCPBridge:
    """Create a WorkstationMCPBridge instance isolated inside a temporary repo root."""
    persist_path = tmp_path / "chroma_test_db"
    return WorkstationMCPBridge(
        repo_root=str(tmp_path),
        persist_dir=str(persist_path),
        cpu_ollama_url="http://127.0.0.1:11435",
        embed_model="bge-large",
        worker_model="academic-worker",
        chunk_size=100,
        chunk_overlap=20,
    )


def test_bridge_init_predefined_attributes(tmp_path: Path) -> None:
    """Verify __init__ assigns all predefined configuration and lazy None attributes."""
    persist_dir = str(tmp_path / "db")
    instance = WorkstationMCPBridge(repo_root=str(tmp_path), persist_dir=persist_dir)

    assert instance.repo_root == tmp_path.resolve()
    assert instance.persist_dir == persist_dir
    assert instance.cpu_ollama_url == "http://127.0.0.1:11435"
    assert instance.embed_model == "bge-large"
    assert instance.worker_model == "academic-worker"
    assert instance.chunk_size == 1200
    assert instance.chunk_overlap == 200
    assert instance.vector_db is None
    assert instance.text_splitter is None


def test_resolve_target_path_valid_relative_and_absolute(
    bridge: WorkstationMCPBridge, tmp_path: Path
) -> None:
    """Verify valid relative and absolute paths inside repo_root resolve accurately."""
    sub_file = tmp_path / "AI" / "sample.py"
    sub_file.parent.mkdir(parents=True, exist_ok=True)
    sub_file.write_text("x = 1\n", encoding="utf-8")

    assert bridge._resolve_target_path("AI/sample.py") == sub_file.resolve()
    assert bridge._resolve_target_path(str(sub_file)) == sub_file.resolve()
    assert bridge._resolve_target_path(".") == tmp_path.resolve()


def test_resolve_target_path_rejects_outside_repo_root(
    bridge: WorkstationMCPBridge, tmp_path: Path
) -> None:
    """Verify path traversal or external system paths raise a ValueError."""
    outside_path = tmp_path.parent / "outside_secret.py"
    with pytest.raises(ValueError, match="outside repository root"):
        bridge._resolve_target_path(str(outside_path))

    with pytest.raises(ValueError, match="outside repository root"):
        bridge._resolve_target_path("../outside_secret.py")


def test_collect_files_filters_extensions_and_ignored_dirs(
    bridge: WorkstationMCPBridge, tmp_path: Path
) -> None:
    """Verify _collect_files includes allowed extensions and skips ignored directories."""
    valid_py = tmp_path / "AI" / "worker.py"
    valid_md = tmp_path / "README.md"
    ignored_ext = tmp_path / "binary.bin"
    ignored_dir_file = tmp_path / "__pycache__" / "cached.py"

    valid_py.parent.mkdir(parents=True, exist_ok=True)
    ignored_dir_file.parent.mkdir(parents=True, exist_ok=True)

    valid_py.write_text("print('ok')\n", encoding="utf-8")
    valid_md.write_text("# Docs\n", encoding="utf-8")
    ignored_ext.write_bytes(b"\x00\x01")
    ignored_dir_file.write_text("print('skip')\n", encoding="utf-8")

    collected = bridge._collect_files(tmp_path)
    assert collected == [valid_py.resolve(), valid_md.resolve()]


def test_ingest_path_missing_target(bridge: WorkstationMCPBridge) -> None:
    """Verify ingest_path returns a clear message when target path does not exist."""
    result = bridge.ingest_path("nonexistent/folder")
    assert "Target path does not exist" in result


def test_ingest_path_chunks_and_deduplicates(
    bridge: WorkstationMCPBridge, tmp_path: Path
) -> None:
    """Verify ingest_path deletes existing chunks when replace_existing=True and adds new chunks."""
    target_file = tmp_path / "AI" / "module.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("def alpha():\n    return 'hello world' * 10\n", encoding="utf-8")

    mock_db = MagicMock()
    bridge.vector_db = mock_db

    result = bridge.ingest_path("AI/module.py", replace_existing=True)

    assert "Ingestion complete:" in result
    assert "1 file(s)" in result
    mock_db.delete.assert_called_once_with(where={"source": "AI/module.py"})
    mock_db.add_texts.assert_called_once()
    call_kwargs = mock_db.add_texts.call_args.kwargs
    assert len(call_kwargs["texts"]) >= 1
    assert call_kwargs["metadatas"][0]["source"] == "AI/module.py"


def test_query_local_rag_returns_formatted_documents(
    bridge: WorkstationMCPBridge, tmp_path: Path
) -> None:
    """Verify query_local_rag formats matching ChromaDB documents with source headers."""
    Path(bridge.persist_dir).mkdir(parents=True, exist_ok=True)
    mock_db = MagicMock()
    mock_db.similarity_search.return_value = [
        Document(page_content="class MultiAgent:", metadata={"source": "AI/multiAgent.py"}),
        Document(page_content="PARAMETER num_ctx 24576", metadata={"source": "AI/gpu.modelfile"}),
    ]
    bridge.vector_db = mock_db

    output = bridge.query_local_rag("MultiAgent", top_k=2)

    mock_db.similarity_search.assert_called_once_with("MultiAgent", k=2)
    assert "[Source: AI/multiAgent.py]\nclass MultiAgent:" in output
    assert "[Source: AI/gpu.modelfile]\nPARAMETER num_ctx 24576" in output


def test_query_local_rag_missing_directory(bridge: WorkstationMCPBridge) -> None:
    """Verify query_local_rag handles uninitialized persist directories gracefully."""
    output = bridge.query_local_rag("anything")
    assert "Vector directory not found" in output


def test_check_gpu_vram_success_and_failure(bridge: WorkstationMCPBridge) -> None:
    """Verify check_gpu_vram parses nvidia-smi output and handles subprocess errors."""
    mock_ok = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="0, NVIDIA GeForce RTX 3060, 1024 MiB, 12288 MiB, 42\n",
        stderr="",
    )
    with patch("subprocess.run", return_value=mock_ok):
        assert "NVIDIA GeForce RTX 3060" in bridge.check_gpu_vram()

    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert "Failed to query nvidia-smi" in bridge.check_gpu_vram()


def test_offload_to_cpu_worker_success_and_network_error(
    bridge: WorkstationMCPBridge,
) -> None:
    """Verify offload_to_cpu_worker parses JSON responses and catches connection errors."""
    fake_body = json.dumps({"response": "Summary complete."}).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = fake_body
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert bridge.offload_to_cpu_worker("Summarize logs") == "Summary complete."

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        assert "CPU Ollama request failed" in bridge.offload_to_cpu_worker("Summarize logs")


def test_mcp_tool_wrappers_delegate_to_bridge() -> None:
    """Verify top-level @mcp.tool functions delegate arguments directly to the bridge."""
    with patch("AI.mcp_workstation_server.bridge") as mock_bridge:
        mock_bridge.ingest_path.return_value = "ingested"
        mock_bridge.query_local_rag.return_value = "results"
        mock_bridge.check_gpu_vram.return_value = "gpu_ok"
        mock_bridge.offload_to_cpu_worker.return_value = "cpu_ok"

        assert ingest_code_to_rag("AI/", replace_existing=False) == "ingested"
        mock_bridge.ingest_path.assert_called_once_with(
            target_path="AI/", replace_existing=False
        )

        assert search_project_rag("query", top_k=3) == "results"
        mock_bridge.query_local_rag.assert_called_once_with(query="query", top_k=3)

        assert get_gpu_vram_status() == "gpu_ok"
        assert summarize_on_cpu_llm("logs") == "cpu_ok"
