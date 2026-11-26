# Repository Workspace & Python Design Standards

## Environment & Architecture
- **Repo Root:** `/home/wrichter/Documents/Code/Projects/Python/processors/`
- **Platform:** Native Ubuntu Linux (bare-metal). Never suggest Dockerfiles, containers, or external cloud API keys.
- **Modular Package Design:** Subfolders (such as `AI/`) are designed as importable Python modules called by root-level scripts. Keep `__init__.py` exports clean and avoid hardcoded relative paths that break when imported from the repo root.
- **File Tool Paths:** Always use paths relative to the PyCharm project root (e.g., `AI/multiAgent.py`). If a relative path fails, list files first to verify the exact path.

## Python Design & PEP 8 Standards
1. **PEP 8 Compliance:** Strictly follow PEP 8 (`snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, 4-space indentation, sorted imports).
2. **Classes & Members:** Encapsulate stateful components and pipelines inside cohesive classes with focused member methods.
3. **Predefined Instance Attributes:** Inside `__init__`, explicitly predefine every instance variable (`self.attribute = None` or a default value) before assigning dynamic state anywhere else.
4. **Stateless Functions:** Use standalone, pure functions instead of classes when no state needs to be maintained across calls.
5. **In-Code Documentation:** Include concise Google-style docstrings for modules, classes, and functions, explicit Python type hints, and brief inline comments for non-obvious logic.
6. **Code Usability:** Prioritize clear naming, sensible default parameters, explicit `try/except` error handling, and clean `if __name__ == "__main__":` blocks so every file can be both imported by root callers and run standalone.
7. **Symbol & Signature Preservation:** Always inspect the actual source file via your read/PSI tools before writing usage examples, tests, or refactors. Strictly preserve existing class names, method names, and function signatures (e.g., `MultiAgent`, `initializeAgents`)—never invent placeholder wrapper classes or fabricated method names unless explicitly asked to rename them.
