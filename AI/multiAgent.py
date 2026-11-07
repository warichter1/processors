"""
    Multi-agent model
    Author:

    # Terminal 1: Start the server on the new port
    OLLAMA_HOST=127.0.0.1:11435 ollama serve
    to Start an agent:
    Pull to the GPU Server (Port 11434):
    OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen2.5-coder:32b-instruct-q4_K_M
    Pull to the CPU Server (Port 11435):
    OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5-coder:14b-instruct-q4_K_M
    Build this directly on your CPU instance profile
    OLLAMA_HOST=127.0.0.1:11435 ollama create cpu-worker -f ./cpu_worker.modelfile
    Academic Worker
    OLLAMA_HOST=127.0.0.1:11435 ollama create academic-worker -f ./academic_worker.modelfile
    GPU Reasoner
    OLLAMA_HOST=127.0.0.1:11434 ollama create gpu-reasoner -f ./gpu_reasoner.modelfile
    CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:11435 ollama serve
    Clear cache before rub:
    ollama stop gpu-reasoner
    ollama stop academic-worker
"""

"""
    Multi-agent model
    Author:

    To Start local environment:
    1. Primary Daemon (Dual GPU - Port 11434):
       sudo systemctl restart ollama
       OLLAMA_HOST=127.0.0.1:11434 ollama create gpu-reasoner -f ./gpu_reasoner.modelfile

    2. Secondary Instance (Ryzen CPU - Port 11435):
       CUDA_VISIBLE_DEVICES="" OLLAMA_HOST=127.0.0.1:11435 ollama serve
       OLLAMA_HOST=127.0.0.1:11435 ollama create cpu-worker -f ./cpu_worker.modelfile
       OLLAMA_HOST=127.0.0.1:11435 ollama create academic-worker -f ./academic_worker.modelfile
"""
import os
import re
import json
import ast
import hashlib
import sqlite3
import subprocess
import logging
import warnings
import uuid
import argparse
import requests
import sys
from datetime import datetime
import time
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain_openai import ChatOpenAI

# from AI.runDebugCrew import MultiAgentDebugWorkstation

# 1. Block the string-matching character approximation warning completely
warnings.filterwarnings("ignore", message=".*approximate input token count.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Quiet down the internal CrewAI telemetry log chatters
logging.getLogger("crewai").setLevel(logging.ERROR)

ARCHITECT_TASK_PROMPT = (
    "Review the project gaps compiled by the research fellow in the previous step.\n"
    "Design and output an advanced version of 'output_system.py' that acts as an automated "
    "file operations monitor.\n\n"
    "███████ REQUIRED ENHANCEMENTS & MONITORING HOOKS ███████\n"
    "1. Self-Contained Logging: Implement a logger that automatically appends entries to a local file named 'ai.log'.\n"
    "2. Automated File Tracking: Implement function hooks that safely log file system operations (Create, Read, Update, Delete) "
    "occurring inside the workspace directory.\n"
    "3. Performance Timestamps: Every file tracking action MUST measure and log the precise duration of the operation. "
    "Use high-resolution tracking (e.g., time.perf_counter() or time.perf_counter_ns()) to determine potential script optimizations.\n"
    "4. Active Output verification: Your __main__ execution block MUST run sample tracking calls (e.g., creating a mock data file, "
    "reading it, updating it, and deleting it) and print explicit step markers to stdout so the compiler sees immediate verification logs.\n\n"
    "███████ CRITICAL SYSTEM GUARDRAILS ███████\n"
    "A. NEVER use logging.config.fileConfig('logging.conf') or reference external configuration templates.\n"
    "B. You must wrap your complete python script inside a clean markdown block exactly like this:\n"
    "```python\n"
    "# Your complete enhanced script code goes here\n"
    "```\n"
    "Provide nothing but the valid python code block inside your response."
)
RESEARCH_TASK_PROMPT = (
    "Analyze this list of existing files inside your target directory path:\n"
    "Path: {target_dir}\n"
    "Discovered Files: [{files}]\n\n"
    "Compile a brief markdown summary detailing what configuration structures are missing "
    "to deploy a robust logging baseline system."
)
RESEARCH_MODEL = "ollama/academic-worker"
ARCHITECT_MODEL = "ollama/gpu-reasoner"
CACHE_SCHEMA_VERSION = 1

CONCISE_ARCHITECT_TASK_PROMPT = (
    "Using the preceding research findings, generate output_system.py as a self-contained Python file-operation monitor. "
    "Implement safe create, read, update, and delete functions confined to a workspace; log each operation's duration "
    "with time.perf_counter() to ai.log. Configure logging entirely in code: do not use fileConfig or external configs. "
    "In __main__, exercise all four operations and print a step marker for each. Return only the complete script in one "
    "```python code fence."
)



class MultiAgent:
    """Multi-agent model leveraging native file execution bypass strings"""

    def __init__(self, targetDir="/home/wrichter/Documents/Code/Projects/Python/processors/AI", debug=True):
        print("Initializing Multi-agent environment configuration...")
        self.TARGET_DIR = targetDir
        os.makedirs(self.TARGET_DIR, exist_ok=True)
        self.starttime = time.time()
        self.log_dir = os.path.join(self.TARGET_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.cache_path = os.path.join(self.log_dir, "multiAgent-cache.sqlite3")
        self.debug = debug
        self.run_id = uuid.uuid4().hex[:12]
        self._run_perf_start = time.perf_counter()
        self._last_time_marker = None
        self._kickoff_started_at = None
        self.profile_logger = None
        self.profile_path = None
        if self.debug:
            self._ensure_profile_logger()
        """
            All tracking, file scanning, and subprocess executions are now handled
            natively via clean Python hooks at the bottom of the execution
        """
        self.agents = {}
        self.tasks = []
        self.my_local_crew = None
        self._agent_step_times = {}
        self._agent_step_baselines = {}
        self._agent_step_counts = {}
        self._agent_tool_counts = {}
        self._task_last_completed_at = None
        self._active_task_name = "setup"
        self._llm_request_count = 0
        self._llm_request_starts = {}
        self._llm_event_handlers = []

    def _log_profile(self, event, **fields):
        if not self.debug:
            return
        self._ensure_profile_logger()
        record = {
            "run_id": self.run_id,
            "event": event,
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.perf_counter() - self._run_perf_start, 6),
            **fields,
        }
        self.profile_logger.info(json.dumps(record, sort_keys=True, default=str))

    def _ensure_profile_logger(self):
        if self.profile_logger is not None:
            return
        self.profile_path = os.path.join(self.log_dir, "multiAgent-profile.jsonl")
        self.profile_logger = logging.getLogger(f"multiAgent.profile.{self.run_id}")
        self.profile_logger.setLevel(logging.INFO)
        self.profile_logger.propagate = False
        profile_handler = logging.FileHandler(
            self.profile_path,
            encoding="utf-8"
        )
        profile_handler.setFormatter(logging.Formatter("%(message)s"))
        self.profile_logger.addHandler(profile_handler)

    def _initialize_cache(self):
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    research_output TEXT NOT NULL,
                    architect_output TEXT NOT NULL,
                    generated_code TEXT NOT NULL
                )
                """
            )

    def _input_file_manifest(self):
        generated_entries = {
            "output_system.py",
            "ai.log",
            "workspace",
            "logs",
            "__pycache__",
        }
        return sorted(
            name for name in os.listdir(self.TARGET_DIR)
            if name not in generated_entries
        )

    def _workflow_cache_inputs(self):
        files = self._input_file_manifest()
        research_prompt = RESEARCH_TASK_PROMPT.format(
            target_dir=self.TARGET_DIR,
            files=", ".join(files) if files else "No files currently exist (Empty directory).",
        )
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "target_dir": self.TARGET_DIR,
            "files": files,
            "research_prompt": research_prompt,
            "architect_prompt": ARCHITECT_TASK_PROMPT,
            "research_model": self._model_cache_identity(
                RESEARCH_MODEL,
                "academic_worker.modelfile"
            ),
            "architect_model": self._model_cache_identity(
                ARCHITECT_MODEL,
                "gpu_reasoner.modelfile"
            ),
            "agent_instructions": {
                "research": {
                    "role": "Senior Academic Research Fellow",
                    "goal": "Examine file system specifications and compile clear structural markdown parameters.",
                    "backstory": "A precise text processing module configured on the local Ryzen CPU layout.",
                    "expected_output": "A structured markdown report reviewing local codebase health.",
                },
                "architect": {
                    "role": "Lead Software Architect",
                    "goal": "Review system layouts and generate robust, ready-to-execute Python scripts.",
                    "backstory": "Advanced code generation matrix utilizing dual-GPU compute engines for clean logical generation.",
                    "expected_output": "A complete, self-contained python script file that logs tracked file operations with performance timestamps.",
                },
            },
        }

    def _model_cache_identity(self, model_name, modelfile_name):
        modelfile_path = os.path.join(self.TARGET_DIR, modelfile_name)
        try:
            with open(modelfile_path, "rb") as modelfile:
                modelfile_hash = hashlib.sha256(modelfile.read()).hexdigest()
        except FileNotFoundError:
            modelfile_hash = None
        return {
            "model": model_name,
            "modelfile_sha256": modelfile_hash,
        }

    @staticmethod
    def _workflow_cache_key(inputs):
        serialized = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_workflow_cache(self, cache_key):
        self._initialize_cache()
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                """
                SELECT inputs_json, research_output, architect_output, generated_code
                FROM workflow_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "inputs_json": row[0],
            "research_output": row[1],
            "architect_output": row[2],
            "generated_code": row[3],
        }

    def _store_workflow_cache(self, cache_key, inputs, research_output, architect_output, generated_code):
        self._initialize_cache()
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                INSERT INTO workflow_cache (
                    cache_key, created_at, inputs_json, research_output, architect_output, generated_code
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    created_at = excluded.created_at,
                    inputs_json = excluded.inputs_json,
                    research_output = excluded.research_output,
                    architect_output = excluded.architect_output,
                    generated_code = excluded.generated_code
                """,
                (
                    cache_key,
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    json.dumps(inputs, sort_keys=True),
                    research_output,
                    architect_output,
                    generated_code,
                ),
            )

    def clear_workflow_cache(self):
        self._initialize_cache()
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("DELETE FROM workflow_cache")

    def getTime(self, text, timestamp=None):
        if not self.debug:
            return
        if timestamp is None:
            timestamp = time.time()
        current_perf = time.perf_counter()
        elapsed = (
            None if self._last_time_marker is None
            else round(current_perf - self._last_time_marker, 6)
        )
        self._last_time_marker = current_perf
        wall_time = datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")
        elapsed_text = "" if elapsed is None else f" (elapsed: {elapsed:.3f}s)"
        print(f"{text}: {wall_time}{elapsed_text}")
        self._log_profile(
            "timestamp",
            label=text,
            wall_time=wall_time,
            elapsed_since_marker_s=elapsed
        )

    def _create_agent_step_callback(self, agent_name):
        def log_agent_step(_step_output):
            now = time.perf_counter()
            previous = self._agent_step_times.get(
                agent_name,
                self._agent_step_baselines.get(agent_name, self._kickoff_started_at)
            )
            self._agent_step_times[agent_name] = now
            step_number = self._agent_step_counts.get(agent_name, 0) + 1
            self._agent_step_counts[agent_name] = step_number

            tool_name = None
            if isinstance(_step_output, dict):
                candidate = _step_output.get("tool_name", _step_output.get("tool"))
                tool_name = candidate if isinstance(candidate, str) else None
            else:
                candidate = getattr(
                    _step_output,
                    "tool_name",
                    getattr(_step_output, "tool", None)
                )
                tool_name = candidate if isinstance(candidate, str) else None
            if tool_name:
                self._agent_tool_counts[agent_name] = self._agent_tool_counts.get(agent_name, 0) + 1

            interval = None if previous is None else round(now - previous, 6)
            self.getTime(f"{agent_name} Step {step_number}")
            self._log_profile(
                "agent_step",
                agent=agent_name,
                step=step_number,
                interval_since_previous_step_s=interval,
                output_type=type(_step_output).__name__,
                tool_name=tool_name,
            )

        return log_agent_step

    def _create_task_callback(self, task_name, next_agent=None):
        def log_task_complete(task_output):
            now = time.perf_counter()
            started_at = self._task_last_completed_at
            self._task_last_completed_at = now
            duration = round(now - started_at, 6) if started_at is not None else None
            if next_agent:
                self._agent_step_baselines[next_agent] = now
                self._active_task_name = "Architect"
            self.getTime(f"{task_name} Task Complete")

            raw_output = getattr(task_output, "raw", "") or ""
            agent = getattr(task_output, "agent", None)
            self._log_profile(
                "task_complete",
                task=task_name,
                task_duration_s=duration,
                duration_basis="kickoff_or_previous_task_callback_to_this_task_callback",
                output_chars=len(raw_output),
                agent=str(agent) if agent else None,
            )

        return log_task_complete

    def install_llm_request_metrics(self):
        """Record request latency and token usage from CrewAI LLM lifecycle events."""
        if self._llm_event_handlers:
            return
        try:
            from crewai.events import crewai_event_bus
            from crewai.events.types.llm_events import (
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
                LLMCallStartedEvent,
            )
        except ImportError as exc:
            raise RuntimeError(
                "CrewAI LLM lifecycle events are unavailable; cannot measure request latency."
            ) from exc

        workstation = self

        def prompt_character_count(messages):
            if isinstance(messages, str):
                return len(messages)
            if not isinstance(messages, list):
                return 0
            return sum(
                len(str(message.get("content", "")))
                for message in messages
                if isinstance(message, dict)
            )

        def record_request(event, outcome, error=None):
            started_at = workstation._llm_request_starts.pop(event.call_id, None)
            duration = None
            if started_at is not None:
                try:
                    duration = max(0.0, (event.timestamp - started_at).total_seconds())
                except (AttributeError, TypeError):
                    duration = None

            usage = getattr(event, "usage", None)
            if not isinstance(usage, dict):
                usage = {}
            completion_tokens = usage.get("completion_tokens")
            tokens_per_second = (
                workstation._tokens_per_second(completion_tokens, duration * 1_000_000_000)
                if duration else None
            )
            messages = getattr(event, "messages", None)
            workstation._llm_request_count += 1
            workstation._log_profile(
                "llm_request",
                request_index=workstation._llm_request_count,
                request_event=outcome,
                task=getattr(event, "task_name", None) or self._active_task_name,
                agent=getattr(event, "agent_role", None),
                model=getattr(event, "model", None),
                duration_s=round(duration, 6) if duration is not None else None,
                prompt_chars=prompt_character_count(messages),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens"),
                completion_tokens_per_second=tokens_per_second,
                finish_reason=getattr(event, "finish_reason", None),
                response_id=getattr(event, "response_id", None),
                error=error,
            )

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(_source, event):
            workstation._llm_request_starts[event.call_id] = event.timestamp

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(_source, event):
            record_request(event, "completed")

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(_source, event):
            record_request(event, "failed", error=event.error)

        self._llm_event_handlers = [
            on_llm_call_started,
            on_llm_call_completed,
            on_llm_call_failed,
        ]
        self._log_profile("llm_request_metrics_enabled", callback="CrewAI event bus")

    @staticmethod
    def _nanoseconds_to_seconds(value):
        if isinstance(value, (int, float)) and value > 0:
            return round(value / 1_000_000_000, 6)
        return None

    @staticmethod
    def _tokens_per_second(token_count, duration_ns):
        if (
            isinstance(token_count, (int, float))
            and isinstance(duration_ns, (int, float))
            and token_count > 0
            and duration_ns > 0
        ):
            return round(token_count / (duration_ns / 1_000_000_000), 3)
        return None

    def benchmark_ollama(self, repetitions=1, max_tokens=1024, selected_models=None):
        """Compare candidate Architect models and prompt variants via Ollama metrics."""
        research_context = (
            "\n\nResearch findings: Keep the monitor self-contained, use a local workspace "
            "directory, and include a concise smoke test."
        )
        benchmark_prompts = {
            "current": ARCHITECT_TASK_PROMPT + research_context,
            "concise": CONCISE_ARCHITECT_TASK_PROMPT + research_context,
        }
        candidates = selected_models or [
            ("gpu-reasoner", "http://127.0.0.1:11434"),
            ("qwen2.5-coder:32b-instruct-q4_K_M", "http://127.0.0.1:11434"),
            ("cpu-worker", "http://127.0.0.1:11435"),
        ]
        if repetitions < 1:
            raise ValueError("Benchmark repetitions must be at least 1.")
        if max_tokens < 1:
            raise ValueError("Benchmark max tokens must be at least 1.")

        self._log_profile(
            "ollama_benchmark_started",
            repetitions=repetitions,
            max_tokens=max_tokens,
            model_candidates=[{"model": model, "endpoint": url} for model, url in candidates],
            prompt_variants=list(benchmark_prompts),
        )
        for model, base_url in candidates:
            for prompt_name, prompt in benchmark_prompts.items():
                for repetition in range(1, repetitions + 1):
                    started = time.perf_counter()
                    try:
                        response = requests.post(
                            f"{base_url}/api/generate",
                            json={
                                "model": model,
                                "prompt": prompt,
                                "stream": False,
                                "keep_alive": "10m",
                                "options": {"temperature": 0.2, "num_predict": max_tokens},
                            },
                            timeout=(5, 1200),
                        )
                        response.raise_for_status()
                        result = response.json()
                    except (requests.RequestException, ValueError) as exc:
                        self._log_profile(
                            "ollama_benchmark",
                            model=model,
                            endpoint=base_url,
                            prompt_variant=prompt_name,
                            prompt_chars=len(prompt),
                            repetition=repetition,
                            success=False,
                            wall_duration_s=round(time.perf_counter() - started, 6),
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        print(
                            f"[BENCHMARK] model={model}, prompt={prompt_name}, "
                            f"run={repetition}, failed={type(exc).__name__}: {exc}"
                        )
                        continue

                    generated = result.get("response", "")
                    code_match = re.search(r"```python\s*(.*?)```", generated, re.DOTALL)
                    generated_code = code_match.group(1).strip() if code_match else ""
                    syntax_valid = False
                    syntax_error = None
                    operation_functions = []
                    if generated_code:
                        try:
                            syntax_tree = ast.parse(generated_code)
                            syntax_valid = True
                            function_names = {
                                node.name.lower()
                                for node in ast.walk(syntax_tree)
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            }
                            operation_functions = [
                                operation
                                for operation in ("create", "read", "update", "delete")
                                if any(operation in function_name for function_name in function_names)
                            ]
                        except SyntaxError as exc:
                            syntax_error = f"{exc.msg} at line {exc.lineno}"
                    eval_count = result.get("eval_count")
                    eval_duration = result.get("eval_duration")
                    prompt_eval_count = result.get("prompt_eval_count")
                    prompt_eval_duration = result.get("prompt_eval_duration")
                    wall_duration = round(time.perf_counter() - started, 6)
                    metrics = {
                        "wall_duration_s": wall_duration,
                        "total_duration_ns": result.get("total_duration"),
                        "total_duration_s": self._nanoseconds_to_seconds(result.get("total_duration")),
                        "load_duration_ns": result.get("load_duration"),
                        "load_duration_s": self._nanoseconds_to_seconds(result.get("load_duration")),
                        "prompt_eval_count": prompt_eval_count,
                        "prompt_eval_duration_ns": prompt_eval_duration,
                        "prompt_eval_duration_s": self._nanoseconds_to_seconds(prompt_eval_duration),
                        "prompt_tokens_per_second": self._tokens_per_second(
                            prompt_eval_count,
                            prompt_eval_duration
                        ),
                        "eval_count": eval_count,
                        "eval_duration_ns": eval_duration,
                        "eval_duration_s": self._nanoseconds_to_seconds(eval_duration),
                        "generation_tokens_per_second": self._tokens_per_second(eval_count, eval_duration),
                        "output_chars": len(generated),
                        "has_python_code_fence": "```python" in generated,
                        "generated_code_chars": len(generated_code),
                        "python_syntax_valid": syntax_valid,
                        "syntax_error": syntax_error,
                        "contains_main_guard": "__main__" in generated_code,
                        "tracked_operation_functions": operation_functions,
                        "has_file_config": "fileConfig(" in generated,
                    }
                    self._log_profile(
                        "ollama_benchmark",
                        model=model,
                        endpoint=base_url,
                        prompt_variant=prompt_name,
                        prompt_chars=len(prompt),
                        repetition=repetition,
                        success=True,
                        **metrics,
                    )
                    print(
                        f"[BENCHMARK] model={model}, prompt={prompt_name}, "
                        f"run={repetition}, wall={wall_duration:.3f}s, "
                        f"load={metrics['load_duration_s']}s, "
                        f"prompt_eval={metrics['prompt_eval_duration_s']}s, "
                        f"generation={metrics['eval_duration_s']}s, "
                        f"tokens_per_second={metrics['generation_tokens_per_second']}, "
                        f"code_fence={metrics['has_python_code_fence']}"
                    )
        self._log_profile("ollama_benchmark_finished")

    def initializeAgents(self):
        """Sequential hardware and agent bootstrap lifecycle"""
        print("Configuring local hardware execution tiers...")

        # Port 11434 = Dual GPU Instance Profile (32B Quantized Model)
        self.getTime('LLM Configuration Start')
        gpu_reasoner_llm = LLM(model=ARCHITECT_MODEL, base_url="http://127.0.0.1:11434")
        # Port 11435 = 16-Thread Ryzen CPU Profile (14B Quantized Model)
        cpu_academic_llm = LLM(model=RESEARCH_MODEL, base_url="http://127.0.0.1:11435")
        self.getTime('LLM Configuration Complete')

        # 1. Define crisp, lightweight personas without complex framework tools parameters
        self.getTime('Agent Construction Start')
        self.agents['research_agent'] = Agent(
            role="Senior Academic Research Fellow",
            goal="Examine file system specifications and compile clear structural markdown parameters.",
            backstory="A precise text processing module configured on the local Ryzen CPU layout.",
            llm=cpu_academic_llm,
            allow_delegation=False,
            verbose=self.debug,
            step_callback=self._create_agent_step_callback("research_agent")
        )

        self.agents['lead_architect'] = Agent(
            role="Lead Software Architect",
            goal="Review system layouts and generate robust, ready-to-execute Python scripts.",
            backstory="Advanced code generation matrix utilizing dual-GPU compute engines for clean logical generation.",
            llm=gpu_reasoner_llm,
            allow_delegation=False,
            verbose=self.debug,
            step_callback=self._create_agent_step_callback("lead_architect")
        )
        self.getTime('Agent Construction Complete')

        # 2. Build explicit workflow tasks
        # Pass your directory file list directly into the task context string to bypass the list_dir loop entirely
        self.getTime('Task Construction Start')
        existing_files = self._input_file_manifest()
        files_str = ", ".join(existing_files) if existing_files else "No files currently exist (Empty directory)."

        task_research = Task(
            description=RESEARCH_TASK_PROMPT.format(
                target_dir=self.TARGET_DIR,
                files=files_str,
            ),
            expected_output="A structured markdown report reviewing local codebase health.",
            agent=self.agents['research_agent'],
            callback=self._create_task_callback("Research", next_agent="lead_architect")
        )

        # Task 2: Hard-constrained to generate self-contained, high-fidelity monitoring code
        task_architect_build = Task(
            description=ARCHITECT_TASK_PROMPT,
            expected_output="A complete, self-contained python script file that logs tracked file operations with performance timestamps.",
            agent=self.agents['lead_architect'],
            callback=self._create_task_callback("Architect")
        )
        self.tasks.extend([task_research, task_architect_build])
        self.getTime('Task Construction Complete')

        print("Constructing multi-agent crew execution framework...")
        self.getTime('Crew Construction Start')
        self.my_local_crew = Crew(
            agents=[self.agents['research_agent'], self.agents['lead_architect']],
            tasks=self.tasks,
            process=Process.sequential
        )
        self.getTime('Crew Construction Complete')
        print('Workstation Initialization Complete.')

    def initialize_local_rag(self):
        """Loads bge-large embeddings locally using your Ollama CPU endpoint on port 11435"""
        print("Initializing 100% offline local semantic memory layer via Ollama...")

        # This completely isolates your vector transformations to your local loopback address
        local_embeddings = OllamaEmbeddings(
            base_url="http://127.0.0.1:11435",
            model="bge-large"
        )

        # Build your local system vector DB using the offline embeddings layer
        self.vector_db = Chroma(embedding_function=local_embeddings)

        # Seed local knowledge array rules
        self.vector_db.add_texts(
            texts=[
                "Rule: Scripts must encapsulate execution points inside main block handlers with terminal status tracking."],
            metadatas=[{"source": "workstation_policy.txt"}]
        )

    def register_custom_tools(self):
        """
           Defines explicit tools inside class space to bind with personas
           We enforce precise, easy-to-parse lowercase string names for the tools
        """
        @tool("query_rag")
        def query_rag_tool(query: str) -> str:
            """Searches local high-fidelity vector memory pools for past structural files or coding constraints."""
            docs = self.vector_db.similarity_search(query, k=3)
            return "\n---\n".join([d.page_content for d in docs]) if docs else "No structural records found."

        @tool("write_file")
        def file_writer_tool(filename: str, content: str) -> str:
            """Writes fresh scripts directly to the disk environment and auto-indexes text strings into CPU memory."""
            self.isDebug(f"Tool Write Target Action: {filename}")
            safe_path = os.path.join(self.TARGET_DIR, os.path.basename(filename))
            try:
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.vector_db.add_texts(texts=[content], metadatas=[{"source": filename}])
                return f"Success: File written cleanly to {safe_path} and synced to vector storage memory."
            except Exception as e:
                self.log_failure("write_file", "Valid file write IO stream", str(e),
                                 "Local file system permission crash.")
                return f"System file fault error: {str(e)}"

        @tool("execute_script")
        def python_execution_tool(filename: str) -> str:
            """Executes generated python files natively on Ubuntu terminal layer, reporting outputs or traceback bugs."""
            safe_path = os.path.join(self.TARGET_DIR, os.path.basename(filename))
            try:
                result = subprocess.run(["python3", safe_path], capture_output=True, text=True, timeout=20,
                                        cwd=self.TARGET_DIR)
                return f"Success Code 0:\n{result.stdout}" if result.returncode == 0 else f"Failed Code {result.returncode}:\n{result.stderr}"
            except Exception as e:
                self.log_failure("execute_script", "Subprocess exit code check", str(e),
                                 "Script processing runtime exception.")
                return f"Execution shell processing timeout fault: {str(e)}"

        # Enforce clean names on the native pre-built tools as well by instantiating them cleanly
        self.directory_scanner_tool = DirectoryReadTool(directory=self.TARGET_DIR, name="list_dir")
        self.file_reader_tool = FileReadTool(name="read_file")

        self.query_rag_tool = query_rag_tool
        self.file_writer_tool = file_writer_tool
        self.python_execution_tool = python_execution_tool

    def configRegistry(self):
        """Configure agent endpoints and map target compute tiers"""
        print("Configuring LLM communication endpoints...")
        # Define a basic string token split counter function
        gpu_reasoner_llm = LLM(
            model="ollama/gpu-reasoner",
            base_url="http://127.0.0.1:11434",
            stop=["\nObservation:", "Observation:", "Task:"]  # Prevents generating trailing text blocks
        )
        cpu_academic_llm = LLM(
            model="ollama/academic-worker",
            base_url="http://127.0.0.1:11435",
            stop=["\nObservation:", "Observation:", "Task:"]  # Prevents generating trailing text blocks
        )
        gpu_llm = LLM(
            model="ollama/qwen2.5-coder:32b-instruct-q4_K_M",
            base_url="http://127.0.0.1:11434"
        )
        cpu_llm = LLM(
            model="ollama/cpu-worker",
            base_url="http://127.0.0.1:11435"
        )
        gpu_critic_llm = LLM(
            model="ollama/qwen2.5-coder:32b-instruct-q4_K_M",
            base_url="http://127.0.0.1:11434",
            temperature=0.2
        )
        print("Mapping specialization tool boundaries to persona profiles...")
        self.PERSONA_REGISTRY = {
            "lead_architect": {
                "role": "Lead Software Architect",
                "goal": "Synthesize messy project configurations and deploy finalized script designs.",
                "backstory": "Advanced structural intelligence utilizing dual-GPU compute matrices for system execution paths.",
                "llm": gpu_reasoner_llm,
                "tools": [self.file_writer_tool, self.python_execution_tool]
            },
            "research_agent": {
                "role": "Senior Academic Research Fellow",
                "goal": "Scan files, manage code block lookup streams, and extract knowledge markers precisely.",
                "backstory": "High-accuracy text processing model assigned to parsing operations on the Ryzen CPU framework.",
                "llm": cpu_academic_llm,
                "tools": [self.directory_scanner_tool, self.file_reader_tool, self.query_rag_tool]
            },
            "academic_researcher": {
                "role": "Senior Academic Research Expert",
                "goal": "Conduct deep structural data lookups completely devoid of subjective content.",
                "backstory": "Cold logic engine processing massive textual clusters out of System RAM nodes.",
                "temperature": 0.2,
                "llm": cpu_llm,
                "tools": [self.directory_scanner_tool, self.file_reader_tool, self.query_rag_tool]
            },
            "technical_writer": {
                "role": "Lead Systems Documentation Specialist",
                "goal": "Translate code structures into clean, beautifully organized markdown documentation guides.",
                "backstory": "Communication master focused on formatting and clear information headers.",
                "temperature": 0.7,
                "llm": gpu_llm,
                "tools": [self.file_reader_tool]
            }
        }

    def isDebug(self, text):
        if self.debug:
            print(text)

    def create_persona_agent(self, allow_delegation: bool = False) -> Agent:
        """Factory function to build real validated Agent instances from registry blueprints"""
        print("Instantiating validated CrewAI Agent instances...")

        def local_token_counter(text: str) -> int:
            return len(text.split())

        for persona_key, config in self.PERSONA_REGISTRY.items():
            print(f"\tConstructing Agent Object: {persona_key}")

            # Force the model to strip markdown fences and stop JSON loops natively
            raw_system_prompt = (
                f"You are a {config['role']}. {config['backstory']}\n"
                "CRITICAL: Do NOT wrap your tools inside code blocks or markdown backticks (```). "
                "Do NOT write JSON structures like '{\"name\": ...}'. You must output raw plaintext lines starting exactly with 'Thought:', 'Action:', and 'Action Input:'."
            )
            agent_kwargs = {
                "role": config["role"],
                "goal": config["goal"],
                "backstory": config["backstory"],
                "llm": config["llm"],
                "allow_delegation": False,
                "verbose": self.debug,
                "tools": config["tools"],
                "token_counter": local_token_counter,
                "max_iter": 12,
                "cache": False,
                "system_template": raw_system_prompt,
                "step_callback": self._create_agent_step_callback(persona_key)
            }
            if "temperature" in config:
                agent_kwargs["temperature"] = config["temperature"]

            # Instantiate the verified core agent wrapper
            self.agents[persona_key] = Agent(**agent_kwargs)

    def workflowTask(self):
        """Assembles explicit sequential script pipeline tasks using fully initialized objects"""
        print("Assembling optimized system workflow tasks loops...")

        # Task 1: Streamlined to a bulleted gap analysis to prevent token cutoff
        self.getTime('Task Research')
        task_research = Task(
            description=(
                f"Analyze the following list of files inside your directory path:\n"
                f"Path: {self.TARGET_DIR}\n"
                f"Files: [{os.listdir(self.TARGET_DIR)}]\n\n"
                "Output a bullet-point summary identifying exactly what logging scripts are missing. "
                "Do NOT write example code files, config payloads, or long-form documentation blocks."
            ),
            expected_output="A brief, bulleted markdown summary listing missing logging structures.",
            agent=self.agents['research_agent'],
            callback=self._create_task_callback("Research", next_agent="lead_architect")
        )

        # Task 2: Refactored to force zero-dependency configuration setups
        self.getTime('Task Architect Build')
        task_architect_build = Task(
            description=(
                "Review the bulleted gaps reported in the previous step.\n"
                "Write a complete, high-quality, production-ready Python logging script named 'output_system.py'.\n\n"
                "CRITICAL ARCHITECTURAL CONSTRAINTS:\n"
                "1. Do NOT use file-based configurations like logging.config.fileConfig('logging.conf').\n"
                "2. The script must be completely SELF-CONTAINED. Implement logging setup entirely in code using "
                "logging.basicConfig() or logging.config.dictConfig() to output logs directly to a local 'ai.log' file.\n\n"
                "CRITICAL OUTPUT INSTRUCTION:\n"
                "You must wrap your complete python script inside a clean code block like this:\n"
                "```python\n"
                "# Your complete script code goes here\n"
                "```\n"
                "Do not include conversational filler or mock JSON tools."
            ),
            expected_output="A valid python code block containing a ready-to-run system module.",
            agent=self.agents['lead_architect'],
            callback=self._create_task_callback("Architect")
        )

        self.tasks.extend([task_research, task_architect_build])

        print("Constructing multi-agent crew execution framework...")
        self.my_local_crew = Crew(
            agents=[self.agents['research_agent'], self.agents['lead_architect']],
            tasks=self.tasks,
            process=Process.sequential
        )

# Main Native Driver Scope
# Launch the execution pipeline natively on your system
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent workflow or benchmark Ollama models.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare the Architect model candidates using current and concise prompt variants."
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=1,
        help="Number of requests per model/prompt pair (default: 1)."
    )
    parser.add_argument(
        "--benchmark-max-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per benchmark request (default: 1024)."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always run the agents and do not read or write the workflow cache."
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached workflow results before running."
    )
    args = parser.parse_args()

    aiAgents = MultiAgent(debug=True)
    aiAgents._log_profile("run_started", profile_path=aiAgents.profile_path)
    if args.benchmark:
        aiAgents.benchmark_ollama(
            repetitions=args.benchmark_repetitions,
            max_tokens=args.benchmark_max_tokens
        )
        raise SystemExit(0)

    if args.clear_cache:
        aiAgents.clear_workflow_cache()
    cache_inputs = aiAgents._workflow_cache_inputs()
    cache_key = aiAgents._workflow_cache_key(cache_inputs)
    cache_entry = None if args.no_cache else aiAgents._load_workflow_cache(cache_key)
    aiAgents._log_profile(
        "workflow_cache",
        enabled=not args.no_cache,
        hit=cache_entry is not None,
        cache_key=cache_key,
        cache_path=aiAgents.cache_path,
    )
    crew_output = None
    output_text = ""

    if cache_entry is not None:
        print(f"[CACHE] Reusing validated generated work from {aiAgents.cache_path}")
        research_output = cache_entry["research_output"]
        architect_output = cache_entry["architect_output"]
        extracted_code = cache_entry["generated_code"]
    else:
        aiAgents.getTime('Start Time', aiAgents.starttime)
        aiAgents.getTime('Initialize Agents Start')
        aiAgents.initializeAgents()
        aiAgents.install_llm_request_metrics()
        aiAgents.getTime('Initialize Agents Complete')
        aiAgents._agent_step_times.clear()
        aiAgents._agent_step_baselines.clear()
        aiAgents._agent_step_counts.clear()
        aiAgents._agent_tool_counts.clear()
        aiAgents._task_last_completed_at = time.perf_counter()
        kickoff_started_at = time.perf_counter()
        aiAgents._kickoff_started_at = kickoff_started_at
        aiAgents._active_task_name = "Research"
        aiAgents._agent_step_baselines["research_agent"] = kickoff_started_at
        aiAgents.getTime('Crew Kickoff Start')
        try:
            crew_output = aiAgents.my_local_crew.kickoff()
        except Exception as exc:
            kickoff_duration = time.perf_counter() - kickoff_started_at
            aiAgents._log_profile(
                "crew_failed",
                duration_s=round(kickoff_duration, 6),
                error_type=type(exc).__name__,
                error=str(exc),
                agent_steps=aiAgents._agent_step_counts,
                agent_tool_calls=aiAgents._agent_tool_counts,
            )
            raise
        kickoff_duration = time.perf_counter() - kickoff_started_at
        aiAgents.getTime('Crew Kickoff Complete')
        aiAgents._log_profile(
            "crew_complete",
            duration_s=round(kickoff_duration, 6),
            agent_steps=aiAgents._agent_step_counts,
            agent_tool_calls=aiAgents._agent_tool_counts,
        )
        token_usage = getattr(crew_output, "token_usage", None)
        if token_usage is not None:
            usage_fields = (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "successful_requests",
            )
            usage = {
                field: getattr(token_usage, field)
                for field in usage_fields
                if getattr(token_usage, field, None) is not None
            }
            if usage:
                aiAgents._log_profile("token_usage", **usage)
        aiAgents.getTime('Crew Output Conversion Start')
        output_text = str(crew_output)
        aiAgents.getTime('Crew Output Conversion Complete')
        task_outputs = getattr(aiAgents.my_local_crew, "tasks_output", [])
        research_output = str(getattr(task_outputs[0], "raw", "")) if task_outputs else ""
        architect_output = str(getattr(task_outputs[-1], "raw", "")) if task_outputs else output_text
        extracted_code = None

    print("\n" + "=" * 80)
    print("█████████████████████ BARE-METAL AUTOMATED COMPILER █████████████████████")
    print("=" * 80)

    # Gather context strings from text pools
    aiAgents.getTime('Context Collection Start')
    combined_context_pool = [research_output, architect_output] if cache_entry else []
    if cache_entry is None:
        if hasattr(crew_output, 'raw'):
            combined_context_pool.append(str(crew_output.raw))
        combined_context_pool.append(output_text)
        if hasattr(aiAgents.my_local_crew, 'tasks_output'):
            for task_out in aiAgents.my_local_crew.tasks_output:
                combined_context_pool.append(str(task_out.raw))
    aiAgents.getTime('Context Collection Complete')
    aiAgents._log_profile(
        "context_collected",
        block_count=len(combined_context_pool),
        total_chars=sum(len(block) for block in combined_context_pool)
    )
    aiAgents.getTime('Code Extraction Start')
    if cache_entry is None:
        for text_block in combined_context_pool:
            code_block_match = re.search(r"```python\n(.*?)\n```", text_block, re.DOTALL)
            if code_block_match:
                extracted_code = code_block_match.group(1)
                break
    else:
        print("[CACHE] Loaded generated code; skipping model research and generation.")
    aiAgents.getTime('Code Extraction Complete')
    aiAgents._log_profile("code_extracted", output_chars=len(extracted_code) if extracted_code else 0)

    # 2. Hardened In-Memory Patching System
    if extracted_code:
        aiAgents.getTime('Code Hardening Start')
        # Check if the model hallucinated the fileConfig call despite constraints
        if "fileConfig(" in extracted_code:
            print(
                "[!] HARDWARE PARSER WARNING: Hallucinated fileConfig caught in memory. Injecting self-contained array...")

            # Use regex to strip the crashing external file lines out
            extracted_code = re.sub(
                r"logging\.config\.fileConfig\(.*?\)",
                "logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='ai.log', filemode='a')",
                extracted_code
            )

            # Clean up potential dangling config module imports to ensure pristine execution namespaces
            extracted_code = extracted_code.replace("import logging.config", "import logging")
        aiAgents.getTime('Code Hardening Complete')

        script_filename = "output_system.py"
        output_filepath = os.path.join(aiAgents.TARGET_DIR, script_filename)
        print(f"[✓] SUCCESS: Extracted code matrix block. Deploying to: {output_filepath}")
        aiAgents.getTime('Output File Write Start')
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(extracted_code)
        aiAgents.getTime('Output File Write Complete')

        print(f"[!] Launching native subprocess check loop for: {script_filename}...")
        subprocess_started_at = time.perf_counter()
        aiAgents.getTime('Subprocess Validation Start')
        execution_check = None
        try:
            execution_check = subprocess.run(
                [sys.executable, output_filepath],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=aiAgents.TARGET_DIR
            )

            if execution_check.returncode == 0:
                print("\033[92m[✓] EXECUTION SUCCESS (Exit Code 0):\033[0m")
                print(
                    f"--- STDOUT LOGS ---\n{execution_check.stdout if execution_check.stdout else '[Empty Output - Script Executed Silently]'}")
            else:
                print(
                    "\033[91m[X] RUNTIME COMPILATION FAILURE (Exit Code {}):\033[0m".format(execution_check.returncode))
                print(f"--- STDERR ERROR TRACEBACK ---\n{execution_check.stderr}")

        except subprocess.TimeoutExpired:
            print("\033[91m[X] RUNTIME ERROR: Execution process timed out (exceeded 15s sandbox limit).\033[0m")
        finally:
            subprocess_duration = time.perf_counter() - subprocess_started_at
            aiAgents.getTime('Subprocess Validation Complete')
            if execution_check is not None:
                aiAgents._log_profile(
                    "subprocess_complete",
                    duration_s=round(subprocess_duration, 6),
                    exit_code=execution_check.returncode,
                    stdout_chars=len(execution_check.stdout),
                    stderr_chars=len(execution_check.stderr)
                )
            else:
                aiAgents._log_profile(
                    "subprocess_complete",
                    duration_s=round(subprocess_duration, 6),
                    timed_out=True
                )
            if (
                execution_check is not None
                and execution_check.returncode == 0
                and not args.no_cache
                and cache_entry is None
            ):
                aiAgents._store_workflow_cache(
                    cache_key,
                    cache_inputs,
                    research_output,
                    architect_output,
                    extracted_code,
                )
                aiAgents._log_profile(
                    "workflow_cache_stored",
                    cache_key=cache_key,
                    generated_code_chars=len(extracted_code),
                )
    else:
        print(
            "\033[91m[X] PARSING MISMATCH: The local models failed to output a cleanly extractable code block structure.\033[0m")
    endtime = time.time()
    start_dt = datetime.fromtimestamp(aiAgents.starttime)
    end_dt = datetime.fromtimestamp(endtime)
    time_difference = end_dt - start_dt
    aiAgents.getTime('End Time', timestamp=endtime)
    aiAgents._log_profile("run_finished", total_duration_s=round(time_difference.total_seconds(), 6))
    print(f"Total Minutes: {round(time_difference.total_seconds()/60, 3)}")
    print("=" * 80 + "\n")
