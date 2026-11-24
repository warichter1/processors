"""Local MCP server exposing CPU RAG ingestion/search, GPU telemetry, and CPU LLM offloading."""

import json
from pathlib import Path
import subprocess
from typing import List, Optional, Set
import urllib.request

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP


class WorkstationMCPBridge:
    """Encapsulates local Chroma RAG ingestion/search and dual-Ollama hardware tools."""

    ALLOWED_EXTENSIONS: Set[str] = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".modelfile",
    }
    IGNORED_DIRS: Set[str] = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".idea",
        ".devoxxgenie",
        "chromeLangChainDb",
        "chromeLangChainDb1",
        "chromeLangChainDb2",
        "chromeLangChainCodeDb",
    }

    def __init__(
        self,
        repo_root: str = "/home/wrichter/Documents/Code/Projects/Python/processors",
        persist_dir: str = "/home/wrichter/Documents/Code/Projects/Python/processors/AI/chromeLangChainCodeDb",
        cpu_ollama_url: str = "http://127.0.0.1:11435",
        embed_model: str = "bge-large",
        worker_model: str = "academic-worker",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize workstation paths, chunking parameters, and lazy-loaded state."""
        self.repo_root: Path = Path(repo_root).resolve()
        self.persist_dir: str = persist_dir
        self.cpu_ollama_url: str = cpu_ollama_url
        self.embed_model: str = embed_model
        self.worker_model: str = worker_model
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.vector_db: Optional[Chroma] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None

    def _get_vector_db(self) -> Chroma:
        """Lazily initialize the local ChromaDB connection on CPU."""
        if self.vector_db is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            embeddings = OllamaEmbeddings(
                base_url=self.cpu_ollama_url,
                model=self.embed_model,
            )
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
        return self.vector_db

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazily initialize the recursive text splitter."""
        if self.text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self.text_splitter

    def _resolve_target_path(self, target_path: str) -> Path:
        """Resolve a relative or absolute path and restrict it inside repo_root."""
        candidate = Path(target_path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve()

        if resolved != self.repo_root and self.repo_root not in resolved.parents:
            raise ValueError(f"Path '{resolved}' is outside repository root '{self.repo_root}'.")
        return resolved

    def _collect_files(self, target: Path) -> List[Path]:
        """Collect eligible source/doc files from a file or directory path."""
        if target.is_file():
            return [target] if target.suffix.lower() in self.ALLOWED_EXTENSIONS else []

        collected: List[Path] = []
        for item in target.rglob("*"):
            if any(part in self.IGNORED_DIRS for part in item.parts):
                continue
            if item.is_file() and item.suffix.lower() in self.ALLOWED_EXTENSIONS:
                collected.append(item)
        return sorted(collected)

    def ingest_path(self, target_path: str, replace_existing: bool = True) -> str:
        """Chunk and embed a file or directory into local ChromaDB on CPU.

        Args:
            target_path: File or directory path relative to repo root (or absolute within root).
            replace_existing: Delete prior chunks for matching source paths before inserting.

        Returns:
            Summary string detailing indexed files and chunk counts.
        """
        try:
            resolved_target = self._resolve_target_path(target_path)
            if not resolved_target.exists():
                return f"Target path does not exist: {resolved_target}"

            files = self._collect_files(resolved_target)
            if not files:
                return f"No eligible files found at {resolved_target}."

            db = self._get_vector_db()
            splitter = self._get_splitter()

            total_chunks = 0
            indexed_files = 0

            for file_path in files:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    continue

                rel_source = str(file_path.relative_to(self.repo_root))
                if replace_existing:
                    db.delete(where={"source": rel_source})

                chunks = splitter.split_text(content)
                if not chunks:
                    continue

                metadatas = [
                    {"source": rel_source, "chunk_index": idx}
                    for idx in range(len(chunks))
                ]
                db.add_texts(texts=chunks, metadatas=metadatas)
                total_chunks += len(chunks)
                indexed_files += 1

            return (
                f"Ingestion complete: indexed {total_chunks} chunks across "
                f"{indexed_files} file(s) into {self.persist_dir}."
            )
        except Exception as exc:
            return f"Ingestion error: {exc}"

    def query_local_rag(self, query: str, top_k: int = 4) -> str:
        """Search the local Chroma vector database for project code or research notes.

        Args:
            query: Semantic search query string.
            top_k: Number of matching chunks to return.

        Returns:
            Formatted string of matching documents and metadata sources.
        """
        if not Path(self.persist_dir).exists():
            return f"Vector directory not found at {self.persist_dir}."

        try:
            db = self._get_vector_db()
            docs = db.similarity_search(query, k=top_k)
            if not docs:
                return "No matching records found in local ChromaDB."
            results = [
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ]
            return "\n\n---\n\n".join(results)
        except Exception as exc:
            return f"RAG query error: {exc}"

    def check_gpu_vram(self) -> str:
        """Check real-time VRAM usage and temperature on the dual RTX 3060 GPUs.

        Returns:
            Formatted nvidia-smi summary or error message.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except Exception as exc:
            return f"Failed to query nvidia-smi: {exc}"

    def offload_to_cpu_worker(self, prompt: str) -> str:
        """Delegate bulk text summarization or log parsing to the Ryzen CPU Ollama instance.

        Args:
            prompt: Text and instructions to process on port 11435.

        Returns:
            Generated text response from the CPU worker model.
        """
        payload = json.dumps(
            {
                "model": self.worker_model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.cpu_ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as exc:
            return f"CPU Ollama request failed: {exc}"


mcp = FastMCP("Workstation-AI-Bridge")
bridge = WorkstationMCPBridge()


@mcp.tool()
def ingest_code_to_rag(target_path: str, replace_existing: bool = True) -> str:
    """Ingest and embed a project file or directory into local ChromaDB via CPU Ollama."""
    return bridge.ingest_path(target_path=target_path, replace_existing=replace_existing)


@mcp.tool()
def search_project_rag(query: str, top_k: int = 4) -> str:
    """Search local ChromaDB embeddings on CPU for code patterns and project docs."""
    return bridge.query_local_rag(query=query, top_k=top_k)


@mcp.tool()
def get_gpu_vram_status() -> str:
    """Check current VRAM allocation and temperatures across both RTX 3060 GPUs."""
    return bridge.check_gpu_vram()


@mcp.tool()
def summarize_on_cpu_llm(text_payload: str) -> str:
    """Offload heavy text parsing or log summarization to the Ryzen 5950X Ollama server."""
    return bridge.offload_to_cpu_worker(prompt=text_payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
