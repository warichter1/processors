"""Centralized Ollama endpoint and model configuration for the multi-agent system.

All hard-coded loopback addresses, ports, and model names used by
``AI/multiAgent.py`` live here so that deployment topology changes only
require editing this single module.
"""

# --- Ollama instance endpoints -------------------------------------------
OLLAMA_HOST = "127.0.0.1"
OLLAMA_GPU_PORT = 11434   # Dual GPU instance profile (32B quantized model)
OLLAMA_CPU_PORT = 11435   # 16-thread Ryzen CPU profile (14B quantized model)

OLLAMA_GPU_URL = f"http://{OLLAMA_HOST}:{OLLAMA_GPU_PORT}"
OLLAMA_CPU_URL = f"http://{OLLAMA_HOST}:{OLLAMA_CPU_PORT}"

# --- Model names -----------------------------------------------------------
# CrewAI LLM models carry the "ollama/" provider prefix.
GPU_REASONER_MODEL = "ollama/gpu-reasoner"
ACADEMIC_WORKER_MODEL = "ollama/academic-worker"

# Embedding model served by the CPU Ollama instance (no provider prefix).
EMBEDDING_MODEL = "bge-large"
