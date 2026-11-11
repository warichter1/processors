import os
import json
import re
import subprocess
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from crewai_tools import DirectoryReadTool, FileReadTool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class MultiAgentDebugWorkstation:
    """Multi-agent framework with embedded structural debug monitoring"""
    def __init__(self, target_dir="/home/wrichter/Documents/Code/Projects/Python/processors/AI/workspace", debug=True):
        print("Initializing Debug Multi-agent Workstation Profile...")
        self.TARGET_DIR = target_dir
        os.makedirs(self.TARGET_DIR, exist_ok=True)
        self.debug = debug
        self.agents = {}
        self.tasks = []
        self.my_local_crew = None
        self.vector_db = None
        self.failure_log = []

    def isDebug(self, text):
        if self.debug:
            print(f"\033[93m[DEBUG_TRACE] {text}\033[0m")

    def log_failure(self, step_name, expected, actual, reason):
        self.failure_log.append({
            "step_name": step_name,
            "expected_pattern": expected,
            "actual_output": actual,
            "failure_reason": reason
        })
        self.isDebug(f"CRITICAL PARSING MISMATCH RECOGNIZED IN: {step_name}\nReason: {reason}")

    def initialize_and_run(self):
        self.initialize_local_rag()
        self.register_custom_tools()
        self.config_registry()
        self.create_persona_agents()
        self.workflow_tasks()
        result = self.my_local_crew.kickoff()
        print("\n" + "="*60 + "\n--- Final Operational Execution Output ---\n" + str(result))
        self.report_validation_failures()

    def report_validation_failures(self):
        """
        Analyzes self.failure_log data accumulated during the execution run.
        Prints a structured, high-visibility summary table to the Ubuntu terminal.
        """
        print("\n" + "=" * 80)
        print("█████████████████████ SYSTEM VALIDATION REPORT █████████████████████")
        print("=" * 80)

        # If the tracking array is completely empty, the formatting regex successfully hit every hook
        if not hasattr(self, 'failure_log') or len(self.failure_log) == 0:
            print("\033[92m[✓] SUCCESS: No critical ReAct parsing or tool execution anomalies caught.\033[0m")
            print("The local Ollama models successfully aligned with the required title-case regex schemas.")
            print("=" * 80 + "\n")
            return

        total_failures = len(self.failure_log)
        print(f"\033[91m[X] WARNING: Caught {total_failures} local model parsing / orchestration deviations.\033[0m")
        print("-" * 80)

        # Enumerate and print an itemized structural breakdown table
        for index, failure in enumerate(self.failure_log, start=1):
            print(f"Anomaly #{index} | Location: \033[96m{failure.get('step_name', 'Unknown Scope')}\033[0m")
            print(f"  ├─ Reason:    {failure.get('failure_reason', 'N/A')}")
            print(f"  ├─ Expected:  \033[92m{failure.get('expected_pattern', 'N/A')}\033[0m")

            # Truncate clean output logs if they are massive runaway string outputs
            actual_raw = str(failure.get('actual_output', 'N/A'))
            actual_truncated = (actual_raw[:150] + '... [TRUNCATED FOR DISPLAY]') if len(
                actual_raw) > 150 else actual_raw
            print(f"  └─ Real Text: \033[91m{actual_truncated}\033[0m")
            print("-" * 80)

        print("████████████████████████ END OF DIAGNOSTICS ████████████████████████")
        print("=" * 80 + "\n")

    def log_failure(self, step_name: str, expected: str, actual: str, reason: str):
        """Appends verified parsing failures into systemic tracking memory arrays"""
        if not hasattr(self, 'failure_log'):
            self.failure_log = []

        self.failure_log.append({
            "step_name": step_name,
            "expected_pattern": expected,
            "actual_output": actual,
            "failure_reason": reason
        })

        # Direct fallback print notification channel if debug flags are verified
        if getattr(self, 'debug', True):
            print(f"\033[91m[LOGGED EXCEPTION] Validation Fault inside '{step_name}': {reason}\033[0m")

    # (Refer to the complete script structure for tool registrations, step callbacks, and validation reports)
