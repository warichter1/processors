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
external best practices are relevant. Chat requests that explicitly ask for
research or concern changing facts are searched before the response is
generated, including version lists and tables. Research follow-ups reuse the
preceding research question so a request such as “add applications” retains its
product context. The assistant is instructed to distinguish model variants,
quantizations, file sizes, and release versions, reason about claims against the
returned evidence, cite only the supplied `[S#]` sources, and explain which
requested details cannot be verified while still providing supported partial
results. Search results are treated as discovery snippets; the assistant fetches
up to two public source pages (prioritizing links supplied by the user) and
provides bounded page excerpts for verification. Fetched pages are treated as
untrusted text, and private-network, credential-bearing, unsupported, or oversized
URLs/pages are not fetched. Search failures return a clear verification error
instead of silently claiming current facts.
Saved history remains available to the user, but factual research requests do
not send earlier assistant responses back as evidence; follow-ups retain only
the relevant earlier user requests.
Search attempts and outcomes are logged without logging prompt text. Search
results are limited and requests have a timeout.
Do not put credentials, private project content, or personal data in search
queries. Select **Offline** in the web interface to disable web search. The
workflow options can also be supplied directly when constructing `WorkflowRunner`.

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
same chat workflow to other Python entry points. The top menu provides **File**
export, **View** selectable chat width (saved in a browser cookie), **Options** script generation and model benchmarking, **Utilities**
for saved history and ranked research memory, **Sleep** nap/wake controls, and
**Help** page guide and About dialog with the code-file version timestamp,
server start time, and live uptime.
Selecting **Generate Script** or **Benchmark Models** starts the workflow
immediately using the current run settings; results appear in a dialog.
**View Saved History** loads the
transcript into the chat, while **Clear Saved History** deletes it. The
checkboxes control cache and web-search behavior; benchmark repetitions and
token limits are set in the numeric fields. A startup notice appears while the
page initializes and points to the cache and offline settings. Sending a request displays a
processing indicator and temporarily disables Send while making the prompt
read-only until the assistant responds.
Chat turns are stored in `AI/logs/conversation-history.sqlite3` and associated
with the browser's signed session, so they can be restored after a page reload
or server restart. The assistant receives recent conversation context when
relevant; latency records retain overall context size and separately record
prior history and current-prompt sizes for new rows. Older latency rows retain
their combined counts and have zero in the newly added split fields. History
warnings use only prior history.
Clearing history removes that session's transcript and latency records.
Response-time records include the history context size. The UI warns when
context exceeds 12,000 characters, and flags responses over 30 seconds when
large history may have contributed; clear history to reduce the context.
Explicit statements that a prior answer was correct or incorrect update shared,
category-deduplicated research memory in the same SQLite database. A request to
review results records a neutral review event. Ranks are clamped from -20 to 20;
corrections lower the rank, validations raise it, and neutral reviews do not
change it. **View ranked research memory** opens a dialog showing each topic, rank, feedback
counts, and latest lesson. Opening the dialog or running Sleep integrates saved
conversation turns once as neutral, unvalidated candidates. The daily sleep
cycle also recalculates bounded ranks from feedback events. Only explicit
correct/incorrect feedback affects rank or allows a lesson to be recalled for
the assistant persona; unvalidated turns are never treated as evidence. Clearing conversation history also clears that
browser session's short-term memory cache, but does not erase shared long-term
memory. Relevant ranked lessons can be recalled quickly from the session cache
and are passed to the assistant only as fallible feedback—not as factual
evidence or instructions. Current factual claims must still be checked against
fresh search results.
Validate an answer against its citations or other reliable evidence, then say it
was correct/right/accurate or incorrect/wrong and explain any correction.
Validated memories raise their rank; corrections lower it. The Sleep menu's
**Set Scheduled Sleep** dialog persists the daily local start time and maximum
cycle duration. Defaults are 4:00 AM and 30 minutes.
Successfully fetched public-page excerpts are also stored separately with their
source URLs and verification timestamps. Identical URL/excerpt pairs are saved
once and have their verification timestamp refreshed; recent topic-matched
findings may be supplied as discovery leads, never as current evidence, and
findings older than 30 days are not recalled. The web server runs a memory
review cycle daily at the configured time and also offers **Nap** and **Wake** in the Sleep
menu. Sleep re-fetches source pages for findings at least
seven days old, refreshes unchanged findings, and supersedes old excerpts when
the fetched source has changed. Only active findings are recalled; unavailable
pages remain historical rather than being treated as disproven. While sleeping,
the page displays progress and disables Send and microphone; Wake stops after
the current bounded page fetch. If a workflow is active when sleep is requested,
the page displays that the agent is ready to sleep and waiting for the workflow
to finish before it begins checking sources. The download icon beside the
composer buttons exports the latest saved assistant response as a formatted
Word document, retaining Markdown headings, lists, inline emphasis, code, and
tables.
Chat uses a single response agent on the faster academic worker instead of
waiting for the sequential research-and-architect pipeline. Research and
current-information queries are searched directly when web search is enabled;
source counts, fetched-page counts, and search duration are logged without
storing the search query or source URLs. The results are passed to the response agent,
which should cite their `[S#]` IDs and URLs.
Factual tables must cite every row from supplied evidence; the assistant must not
infer a model's intended use, performance, API availability, or capabilities from
its name or parameter count. Recommendations should be identified as
recommendations rather than sourced facts, and related product families must
remain distinct.
The response pipeline rejects researched Markdown tables with missing or unknown
row citations rather than presenting them as verified facts. Hardware-fit
recommendations must state supported quantization and context assumptions or
mark them unknown.
Script generation continues to use both agents.

Web requests are type- and size-validated before reaching the workflow. Backend
exceptions are written to `AI/logs/webserver.log` and clients receive a generic
failure message. Server and profiling logs use a queued single-writer handler
and rotate daily at midnight UTC, retaining at most seven daily files.
Chat model calls have a 120-second timeout and generation calls have a
600-second timeout. The web server rejects a concurrent workflow request after
one second rather than leaving it queued behind an in-progress request. LLM call
start events are recorded in the profiler when debug profiling is enabled; the
web server logs workflow request starts and concurrent-request rejections.
