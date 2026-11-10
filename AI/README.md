AI Agent test area

`multiAgent.py` caches successfully validated generated scripts in
`AI/logs/multiAgent-cache.sqlite3`. The cache key includes the workflow prompts,
input directory listing, model names, and the SHA-256 hashes of the research
and architect Modelfiles. Unchanged workflows reuse cached outputs and skip LLM
calls, but still run the generated script's smoke test.

Use the web page's **Disable cache** option to force a fresh run without reading
or writing cache entries, or **Clear cache first** to remove existing entries.
Cache entries
expire after one hour, including results produced with web research. For the
latest web results, disable the cache.

The Research agent can search public web pages through DuckDuckGo when current
external best practices are relevant. Search results are limited to ten and
requests have a timeout. Do not put credentials, private project content, or
personal data in search queries. Select **Offline** in the web interface to
disable web search. The workflow options can also be supplied directly when
constructing `WorkflowRunner`.

```bash
python AI/multiAgent.py
```

```python
from AI.multiAgent import WorkflowRunner

result = WorkflowRunner(no_cache=True, offline=True).run()
```

## Web chat

Start the Flask-hosted Dash and Plotly interface with:

```bash
python AI/AIServer.py
```

Open `http://127.0.0.1:8050`. Enter a message or use **Use microphone** to
transcribe speech with the browser's Web Speech API; microphone input requires
browser permission and a supported browser. The Flask health endpoint is
available at `/healthz`. The imported `MultiAgentWorkflow` class provides the
same chat workflow to other Python entry points. The **Action** menu selects
chat, script generation, or model benchmarking. The checkboxes control cache
and web-search behavior; benchmark repetitions and token limits are set in the
numeric fields. A startup notice appears while the page initializes and points
to the cache and offline settings. Sending a request displays a spinner until
the assistant completes or reports a generic failure.
Chat uses a single response agent on the faster academic worker instead of
waiting for the sequential research-and-architect pipeline; optional web search
remains available when enabled. Current releases, model catalogs, and other
changing facts should trigger a search without requiring the user to provide a
URL; answers should cite the source URLs. Script generation continues to use
both agents.

Web requests are type- and size-validated before reaching the workflow. Backend
exceptions are written to `AI/logs/webserver.log` and clients receive a generic
failure message. Server and profiling logs use a queued single-writer handler
and rotate daily at midnight UTC, retaining at most seven daily files.
