AI Agent test area

`multiAgent.py` caches successfully validated generated scripts in
`AI/logs/multiAgent-cache.sqlite3`. The cache key includes the workflow prompts,
input directory listing, model names, and the SHA-256 hashes of the research
and architect Modelfiles. Unchanged workflows reuse cached outputs and skip LLM
calls, but still run the generated script's smoke test.

Use `--no-cache` to force a fresh run without reading or writing cache entries,
or `--clear-cache` to remove existing entries before running. For example:

```bash
python AI/multiAgent.py --clear-cache
python AI/multiAgent.py --no-cache
```
