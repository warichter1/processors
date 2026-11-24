---
name: python-refactor-audit
description: Refactors, audits, and verifies Python files for PEP 8 compliance, predefined __init__ variables, Google-style docstrings, usability, and syntax safety using local Ubuntu tools.
---

# Python Refactor & Quality Verification Workflow

When asked to refactor, audit, or finalize Python code, execute the following steps sequentially:

## Step 1: Structural & Design Audit
1. **State Check:** Convert stateless utility classes into standalone functions. Group stateful operations into cohesive classes.
2. **Attribute Predefinition:** Inspect every `__init__` method and ensure all `self.<attr>` variables used across the class are initialized upfront with default values or `None`.
3. **PEP 8 & Naming:** Normalize all identifiers to PEP 8 (`snake_case` / `PascalCase`) and group imports into standard library, third-party, and local blocks.

## Step 2: Documentation & Usability Upgrade
1. **Docstrings & Types:** Add type annotations to all function/method signatures and write clear Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
2. **Standalone Safety:** Wrap executable test/driver code inside an `if __name__ == "__main__":` block with clean error handling and logging.

## Step 3: Local Terminal Verification
Use your bash/terminal tool to validate the modified file inside the active Python environment:
1. Run syntax compilation: `python3 -m py_compile <target_file.py>`
2. If `ruff` or `flake8` is installed in the virtual environment, run `ruff check <target_file.py>` (or `flake8 <target_file.py>`) and resolve any formatting or lint warnings before concluding.

