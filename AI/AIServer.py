"""Flask-hosted Dash interface for the multi-agent assistant."""

from datetime import datetime, timedelta
from io import BytesIO
from time import perf_counter
import math
import os
import re
import threading
import uuid

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from docx import Document
from docx.shared import Inches
from flask import Flask, jsonify, session
import plotly.graph_objects as go
from werkzeug.exceptions import HTTPException

if __package__:
    from .conversation_history import ConversationHistory
    from .logging_config import get_queued_logger
    from .multiAgent import MultiAgentWorkflow, WorkflowRunner
    from .request_validation import (
        MAX_CONVERSATION_CHARS,
        trim_conversation,
        validate_chat_payload,
    )
else:
    from conversation_history import ConversationHistory
    from logging_config import get_queued_logger
    from multiAgent import MultiAgentWorkflow, WorkflowRunner
    from request_validation import (
        MAX_CONVERSATION_CHARS,
        trim_conversation,
        validate_chat_payload,
    )


ALLOWED_MODES = {"chat", "generate", "benchmark"}
ALLOWED_OPTIONS = {"no-cache", "clear-cache", "offline"}
MAX_BENCHMARK_REPETITIONS = 10
MAX_BENCHMARK_TOKENS = 8192
HISTORY_LATENCY_WARNING_CHARS = 12_000
SLOW_RESPONSE_WARNING_SECONDS = 30
REQUEST_LOCK_WAIT_SECONDS = 1
SLEEP_SCHEDULE_INTERVAL_SECONDS = 24 * 60 * 60
SLEEP_SOURCE_AGE_DAYS = 7
SLEEP_MAX_FINDINGS_PER_CYCLE = 25


class AIServer:
    """Create the Flask server and its Dash-based text and speech chat UI."""

    def __init__(self, target_dir=None, debug=False, web_search=True):
        self.server = Flask(__name__)
        self.server.config["MAX_CONTENT_LENGTH"] = 1_048_576
        self.dash_app = Dash(
            __name__,
            server=self.server,
            title="Multi-Agent Assistant",
            update_title="Working...",
        )
        self.workflow = MultiAgentWorkflow(
            target_dir=target_dir,
            debug=debug,
            web_search=web_search,
        )
        log_directory = os.path.join(self.workflow.agent.TARGET_DIR, "logs")
        self.logger = get_queued_logger(
            "AIServer",
            log_directory,
            "webserver.log",
            "%(asctime)s %(levelname)s %(message)s",
        )
        self.workflow.logger = self.logger
        self.history = ConversationHistory(
            os.path.join(log_directory, "conversation-history.sqlite3")
        )
        self.server.secret_key = self.history.get_secret_key()
        self.server.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=365)
        self._request_lock = threading.Lock()
        self._sleep_state_lock = threading.Lock()
        self._sleep_wake_event = threading.Event()
        self._scheduler_stop_event = threading.Event()
        self._sleep_pending = False
        self._sleeping = False
        self._sleep_status = ""
        self._sleep_thread = None
        self._scheduler_thread = None
        self._configure_routes()
        self._configure_layout()
        self._configure_callbacks()

    def _configure_routes(self):
        @self.server.get("/healthz")
        def health_check():
            return jsonify({"status": "ok"})

        @self.server.errorhandler(Exception)
        def handle_unexpected_error(error):
            if isinstance(error, HTTPException):
                return jsonify({"error": "The request could not be completed."}), error.code
            self.logger.exception("Unhandled Flask request error")
            return jsonify({"error": "The request could not be completed."}), 500

    def _configure_layout(self):
        self.dash_app.layout = html.Main(
            [
                html.Header(
                    [
                        html.H1("Multi-Agent Assistant"),
                        html.P("Ask a question by typing or using speech input."),
                    ],
                    className="page-header",
                ),
                html.Div(
                    id="sleep-notice",
                    role="status",
                    **{"aria-live": "polite"},
                ),
                dcc.Interval(
                    id="sleep-state-check",
                    interval=500,
                    n_intervals=0,
                ),
                html.Div(
                    [
                        self._loading_indicator(
                            "Please wait while the web app loads. Use Run settings "
                            "to disable cache, clear cache, or work offline."
                        ),
                        dcc.Interval(
                            id="startup-ready-check",
                            interval=500,
                            max_intervals=1,
                        ),
                    ],
                    id="startup-loading",
                    className="loading-notice",
                    role="status",
                    **{"aria-live": "polite"},
                ),
                html.Section(
                    [
                        html.Div(
                            id="conversation-view",
                            className="conversation-view",
                            children=html.P(
                                "Your conversation will appear here.",
                                className="empty-state",
                            ),
                        ),
                        html.Div(
                            [
                                html.Label("Run settings"),
                                html.Label("Action", htmlFor="workflow-mode"),
                                dcc.Dropdown(
                                    id="workflow-mode",
                                    options=[
                                        {"label": "Chat", "value": "chat"},
                                        {"label": "Generate script", "value": "generate"},
                                        {"label": "Benchmark models", "value": "benchmark"},
                                        {"label": "View saved history", "value": "view-history"},
                                        {"label": "Clear saved history", "value": "clear-history"},
                                        {"label": "View ranked research memory", "value": "view-memory"},
                                        {"label": "Nap (review older memories)", "value": "nap"},
                                        {"label": "Wake from sleep or nap", "value": "wake"},
                                    ],
                                    value="chat",
                                    clearable=False,
                                ),
                                dcc.Checklist(
                                    id="workflow-options",
                                    options=[
                                        {"label": "Disable cache", "value": "no-cache"},
                                        {"label": "Clear cache first", "value": "clear-cache"},
                                        {"label": "Offline (disable web search)", "value": "offline"},
                                    ],
                                    value=[],
                                    className="workflow-options",
                                ),
                                html.Div(
                                    [
                                        html.Label("Benchmark repetitions", htmlFor="benchmark-repetitions"),
                                        dcc.Input(
                                            id="benchmark-repetitions",
                                            type="number",
                                            min=1,
                                            step=1,
                                            value=1,
                                        ),
                                        html.Label("Maximum generated tokens", htmlFor="benchmark-max-tokens"),
                                        dcc.Input(
                                            id="benchmark-max-tokens",
                                            type="number",
                                            min=1,
                                            step=1,
                                            value=1024,
                                        ),
                                    ],
                                    className="benchmark-options",
                                ),
                            ],
                            className="workflow-settings",
                        ),
                        html.Div(
                            [
                                dcc.Textarea(
                                    id="prompt-input",
                                    placeholder="Write a message...",
                                    className="prompt-input",
                                    rows=4,
                                    readOnly=False,
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            html.Span(
                                                "\u2b07",
                                                **{"aria-hidden": "true"},
                                            ),
                                            id="download-button",
                                            n_clicks=0,
                                            type="button",
                                            title="Download latest response as a Word document",
                                            **{
                                                "aria-label": (
                                                    "Download latest response as a Word document"
                                                )
                                            },
                                        ),
                                        html.Button(
                                            "Use microphone",
                                            id="speech-button",
                                            n_clicks=0,
                                            type="button",
                                        ),
                                        html.Button(
                                            "Send",
                                            id="send-button",
                                            n_clicks=0,
                                            type="button",
                                        ),
                                    ],
                                    className="button-row",
                                ),
                                html.Div(id="speech-status", role="status"),
                                dcc.Download(id="response-download"),
                                html.Div(id="download-status", role="status"),
                                dcc.Loading(
                                    html.Div(id="request-status", role="status"),
                                    custom_spinner=self._loading_indicator(
                                        "Searching public sources when needed and preparing your response..."
                                    ),
                                    delay_show=0,
                                    delay_hide=150,
                                    parent_className="request-loading",
                                ),
                            ],
                            className="composer",
                        ),
                    ],
                    className="chat-panel",
                ),
                html.Section(
                    [
                        html.H2("Response time by turn"),
                        dcc.Graph(
                            id="response-chart",
                            figure=self._response_figure([]),
                            config={"displayModeBar": False},
                        ),
                    ],
                    className="chart-panel",
                ),
            ],
            className="app-shell",
        )

    @staticmethod
    def _loading_indicator(message):
        return html.Div(
            [
                html.Span(className="loading-spinner", **{"aria-hidden": "true"}),
                html.Span(message),
            ],
            className="loading-indicator",
        )

    @staticmethod
    def _response_figure(conversation):
        responses = [
            item for item in conversation
            if item.get("role") == "assistant" and item.get("duration_s") is not None
        ]
        figure = go.Figure(
            data=[
                go.Bar(
                    x=[f"Turn {index}" for index in range(1, len(responses) + 1)],
                    y=[item["duration_s"] for item in responses],
                    marker={"color": "#4f46e5"},
                    hovertemplate="%{x}<br>%{y:.2f} seconds<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 15, "b": 40},
            xaxis_title="Conversation turn",
            yaxis_title="Seconds",
            yaxis={"rangemode": "tozero"},
        )
        return figure

    @staticmethod
    def _render_conversation(conversation):
        if not conversation:
            return html.P("Your conversation will appear here.", className="empty-state")

        return [
            html.Article(
                [
                    html.Div(
                        "You" if item["role"] == "user" else "Assistant",
                        className=f"message-label {item['role']}",
                    ),
                    html.Time(
                        item.get("created_at", ""),
                        className="message-time",
                    ),
                    dcc.Markdown(
                        item["content"],
                        className="message-content",
                        dangerously_allow_html=False,
                    ),
                ],
                className=f"message {item['role']}",
                key=f"{index}-{item['role']}",
            )
            for index, item in enumerate(conversation)
        ]

    def _configure_callbacks(self):
        @self.dash_app.callback(
            Output("startup-loading", "style"),
            Input("startup-ready-check", "n_intervals"),
        )
        def hide_startup_notice(_interval_count):
            return {"display": "none"}

        self.dash_app.clientside_callback(
            """
            function(clicks, currentValue) {
                const noUpdate = window.dash_clientside.no_update;
                if (!clicks) {
                    return [noUpdate, noUpdate];
                }
                const SpeechRecognition =
                    window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) {
                    return [noUpdate, "Speech recognition is not supported by this browser."];
                }

                return new Promise(function(resolve) {
                    const recognition = new SpeechRecognition();
                    let finished = false;
                    function finish(value, status) {
                        if (!finished) {
                            finished = true;
                            resolve([value, status]);
                        }
                    }
                    recognition.lang = navigator.language || "en-US";
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;
                    recognition.onresult = function(event) {
                        const transcript = event.results[0][0].transcript;
                        const separator = currentValue && !/\\s$/.test(currentValue) ? " " : "";
                        finish((currentValue || "") + separator + transcript, "Speech captured.");
                    };
                    recognition.onerror = function(event) {
                        finish(noUpdate, "Speech input failed: " + event.error + ".");
                    };
                    recognition.onend = function() {
                        finish(noUpdate, "No speech was detected.");
                    };
                    try {
                        recognition.start();
                    } catch (error) {
                        finish(noUpdate, "Could not start speech recognition: " + error.message);
                    }
                });
            }
            """,
            Output("prompt-input", "value"),
            Output("speech-status", "children"),
            Input("speech-button", "n_clicks"),
            State("prompt-input", "value"),
            prevent_initial_call=True,
        )

        @self.dash_app.callback(
            Output("send-button", "disabled"),
            Output("speech-button", "disabled"),
            Output("sleep-notice", "children"),
            Input("sleep-state-check", "n_intervals"),
        )
        def update_sleep_status(_interval_count):
            return self._sleep_ui_state()

        @self.dash_app.callback(
            Output("response-download", "data"),
            Output("download-status", "children"),
            Input("download-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_latest_response(clicks):
            if not isinstance(clicks, int) or clicks < 1:
                return no_update, no_update
            latest_response = next(
                (
                    item["content"]
                    for item in reversed(
                        self.history.get_messages(self._get_conversation_id())
                    )
                    if item.get("role") == "assistant"
                    and isinstance(item.get("content"), str)
                    and item["content"].strip()
                ),
                None,
            )
            if latest_response is None:
                return no_update, "There is no assistant response to download yet."

            filename = (
                "assistant-response-"
                + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
                + ".docx"
            )
            return (
                dcc.send_bytes(
                    self._build_word_document(latest_response),
                    filename,
                    type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
                "Latest response downloaded as a Word document.",
            )

        @self.dash_app.callback(
            Output("conversation-view", "children"),
            Output("workflow-mode", "value"),
            Output("prompt-input", "value", allow_duplicate=True),
            Output("response-chart", "figure"),
            Output("request-status", "children"),
            Input("send-button", "n_clicks"),
            Input("workflow-mode", "value"),
            Input("startup-ready-check", "n_intervals"),
            State("prompt-input", "value"),
            State("workflow-options", "value"),
            State("benchmark-repetitions", "value"),
            State("benchmark-max-tokens", "value"),
            prevent_initial_call=True,
            running=[
                (Output("prompt-input", "readOnly"), True, False),
            ],
        )
        def send_message(
            _clicks,
            mode,
            _startup_intervals,
            prompt,
            options,
            benchmark_repetitions,
            benchmark_max_tokens,
        ):
            session_id = self._get_conversation_id()
            triggered_properties = ctx.triggered_prop_ids
            triggered_id = ctx.triggered_id
            if (
                "send-button.n_clicks" in triggered_properties
                and mode in ALLOWED_MODES
            ):
                triggered_id = "send-button"
            if triggered_id == "workflow-mode" and mode == "nap":
                started = self._start_sleep("manual nap")
                return (
                    no_update,
                    "chat",
                    no_update,
                    no_update,
                    "The agent is starting a nap."
                    if started
                    else "The agent is already sleeping.",
                )
            if triggered_id == "workflow-mode" and mode == "wake":
                woke = self._wake_sleep()
                return (
                    no_update,
                    "chat",
                    no_update,
                    no_update,
                    "Waking the agent."
                    if woke
                    else "The agent is already awake.",
                )
            if triggered_id == "send-button" and self._is_sleeping():
                return self._safe_callback_result(
                    self.history.get_messages(session_id),
                    "The agent is sleeping. Choose Wake before sending another message.",
                )
            if triggered_id == "workflow-mode" and mode == "clear-history":
                self.history.clear(session_id)
                return (
                    self._render_conversation([]),
                    "chat",
                    no_update,
                    self._response_figure([]),
                    "Saved conversation history cleared.",
                )
            if triggered_id == "workflow-mode" and mode == "view-history":
                saved_history = self.history.get_messages(session_id)
                return (
                    self._render_conversation(trim_conversation(saved_history)),
                    "chat",
                    no_update,
                    self._response_figure(trim_conversation(saved_history)),
                    f"Loaded {len(saved_history)} saved messages.",
                )
            if triggered_id == "workflow-mode" and mode == "view-memory":
                memories = self.history.get_research_memory()
                return (
                    self._render_research_memory(memories),
                    "chat",
                    no_update,
                    self._response_figure([]),
                    f"Loaded {len(memories)} ranked research memories.",
                )
            if triggered_id == "workflow-mode":
                return (no_update,) * 5
            if triggered_id == "startup-ready-check":
                saved_history = self.history.get_messages(session_id)
                status = (
                    f"Restored {len(saved_history)} saved messages."
                    if saved_history
                    else "Conversation history is saved automatically."
                )
                return (
                    self._render_conversation(trim_conversation(saved_history)),
                    no_update,
                    no_update,
                    self._response_figure(trim_conversation(saved_history)),
                    status,
                )
            if triggered_id != "send-button":
                return (no_update,) * 5

            try:
                prompt, _ = validate_chat_payload(prompt, [])
                (
                    mode,
                    options,
                    benchmark_repetitions,
                    benchmark_max_tokens,
                ) = self._validate_workflow_inputs(
                    _clicks,
                    mode,
                    options,
                    benchmark_repetitions,
                    benchmark_max_tokens,
                )
                if mode == "chat" and not prompt:
                    return self._safe_callback_result(
                        [], "Enter or dictate a message before sending."
                    )

            except ValueError:
                self.logger.warning("Rejected invalid web request")
                return self._safe_callback_result(
                    [], "Check the request settings and try again."
                )
            except Exception:
                self.logger.exception("Failed while validating web request")
                return self._safe_callback_result(
                    [], "The assistant could not complete this request."
                )

            started_at = perf_counter()
            saved_history = self.history.get_messages(session_id)
            request_id = uuid.uuid4().hex
            self.logger.info(
                "Workflow request started",
                extra={"request_id": request_id, "mode": mode},
            )
            feedback_outcome = self._research_feedback_outcome(prompt)
            memory_topic = self._previous_research_topic(saved_history)
            memory_update_warning = False
            if feedback_outcome and memory_topic:
                prior_answer = next(
                    (
                        item["content"]
                        for item in reversed(saved_history)
                        if item.get("role") == "assistant"
                    ),
                    "",
                )
                lesson = (
                    f"User marked the previous answer {feedback_outcome}. "
                    f"User feedback: {prompt[:700]}"
                )
                if prior_answer:
                    lesson += (
                        "\nRejected/validated answer excerpt: "
                        + prior_answer[:500]
                    )
                try:
                    self.history.record_research_memory(
                        session_id,
                        memory_topic,
                        lesson,
                        feedback_outcome,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to update ranked research memory",
                        extra={"request_id": request_id},
                    )
                    memory_update_warning = True
            research_memories = (
                self.history.recall_research_memory(
                    session_id,
                    memory_topic if feedback_outcome and memory_topic else prompt,
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                )
                if mode == "chat"
                else []
            )
            factual_results = (
                self.history.recall_source_findings(prompt)
                if mode == "chat"
                and MultiAgentWorkflow._should_search_prompt(prompt)
                else []
            )
            if mode == "chat":
                model_context = trim_conversation(
                    saved_history,
                    max_chars=MAX_CONVERSATION_CHARS - len(prompt),
                )[-12:]
                context_chars = sum(
                    len(item["content"]) for item in model_context
                )
                context_message_count = len(model_context)
                request_context_messages = context_message_count + int(bool(prompt))
                request_context_chars = context_chars + len(prompt)
            else:
                model_context = []
                context_chars = 0
                context_message_count = 0
                request_context_messages = 0
                request_context_chars = 0
            user_message = prompt or (
                "Run script generation." if mode == "generate" else "Run model benchmark."
            )
            workflow_succeeded = False
            if not self._request_lock.acquire(timeout=REQUEST_LOCK_WAIT_SECONDS):
                self.logger.warning(
                    "Workflow request rejected because another request is still running",
                    extra={"request_id": request_id, "mode": mode},
                )
                return self._safe_callback_result(
                    saved_history,
                    "Another request is still running. Please wait for it to finish.",
                )
            try:
                if self._is_sleeping():
                    return self._safe_callback_result(
                        saved_history,
                        "The agent is ready to sleep. Choose Wake before sending another message.",
                    )
                if mode == "chat":
                    answer = self.workflow.respond(
                        prompt,
                        model_context,
                        offline="offline" in options,
                        research_memories=research_memories,
                        factual_results=factual_results,
                        source_finding_callback=self._record_source_findings,
                    )
                else:
                    workflow_result = WorkflowRunner(
                        benchmark=mode == "benchmark",
                        benchmark_repetitions=benchmark_repetitions,
                        benchmark_max_tokens=benchmark_max_tokens,
                        no_cache="no-cache" in options,
                        clear_cache="clear-cache" in options,
                        offline="offline" in options,
                        target_dir=self.workflow.agent.TARGET_DIR,
                        debug=self.workflow.agent.debug,
                    ).run()
                    answer = workflow_result["message"]
                _, validated_response = validate_chat_payload(
                    "", [{"role": "assistant", "content": answer}]
                )
                answer = validated_response[0]["content"]
                workflow_succeeded = True
            except Exception:
                duration = round(perf_counter() - started_at, 3)
                self.logger.exception(
                    "Workflow request failed",
                    extra={"request_id": request_id, "mode": mode},
                )
                answer = "The assistant could not complete this request."
                status = "The assistant could not complete this request."
            else:
                duration = round(perf_counter() - started_at, 3)
                self.logger.info(
                    "Workflow request completed in %.3f seconds "
                    "(history: %d messages, %d characters; current prompt: %d characters)",
                    duration,
                    context_message_count,
                    context_chars,
                    len(prompt),
                    extra={
                        "request_id": request_id,
                        "mode": mode,
                        "history_context_messages": context_message_count,
                        "history_context_chars": context_chars,
                        "conversation_and_prompt_chars": request_context_chars,
                    },
                )
                status = f"Response received in {duration:.2f} seconds."
            finally:
                self._request_lock.release()

            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                if (
                    workflow_succeeded
                    and mode == "chat"
                    and self._is_research_review_request(prompt)
                ):
                    self.history.record_research_memory(
                        session_id,
                        memory_topic or prompt,
                        "A review was requested. Assistant review: " + answer[:1200],
                        "review",
                        created_at,
                    )
            except Exception:
                self.logger.exception(
                    "Failed to update ranked research memory",
                    extra={"request_id": request_id},
                )
                memory_update_warning = True
            if memory_update_warning:
                status += " Warning: ranked research memory could not be updated."

            conversation = list(saved_history)
            conversation.extend(
                [
                    {
                        "role": "user",
                        "content": user_message,
                        "created_at": created_at,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "duration_s": duration,
                        "created_at": created_at,
                    },
                ]
            )
            try:
                self.history.append_turn(
                    session_id,
                    user_message,
                    answer,
                    duration,
                    request_context_messages,
                    request_context_chars,
                    created_at,
                    history_context_messages=context_message_count,
                    history_context_chars=context_chars,
                    prompt_chars=len(prompt),
                )
                conversation = self.history.get_messages(session_id)
            except Exception:
                self.logger.exception(
                    "Failed to save conversation history",
                    extra={"request_id": request_id},
                )
                status += " Warning: this turn could not be saved to history."
            if context_chars >= HISTORY_LATENCY_WARNING_CHARS:
                status += (
                    f" History context is large ({context_chars:,} characters) and may "
                    "increase latency; clear saved history to reduce it."
                )
            if (
                duration >= SLOW_RESPONSE_WARNING_SECONDS
                and context_chars >= HISTORY_LATENCY_WARNING_CHARS
            ):
                status += (
                    f" The {duration:.1f}-second response was slow; the large history "
                    "may have contributed."
                )
            active_conversation = trim_conversation(conversation)
            return (
                self._render_conversation(active_conversation),
                no_update,
                "",
                self._response_figure(active_conversation),
                status,
            )

    @staticmethod
    def _validate_workflow_inputs(
        clicks, mode, options, benchmark_repetitions, benchmark_max_tokens
    ):
        if isinstance(clicks, bool) or not isinstance(clicks, int) or clicks < 1:
            raise ValueError("Invalid request.")
        if not isinstance(mode, str) or mode not in ALLOWED_MODES:
            raise ValueError("Invalid request.")
        if (
            not isinstance(options, list)
            or any(not isinstance(option, str) for option in options)
            or not set(options).issubset(ALLOWED_OPTIONS)
        ):
            raise ValueError("Invalid request.")
        benchmark_repetitions = AIServer._validate_integer(
            benchmark_repetitions, 1, MAX_BENCHMARK_REPETITIONS
        )
        benchmark_max_tokens = AIServer._validate_integer(
            benchmark_max_tokens, 1, MAX_BENCHMARK_TOKENS
        )
        return mode, set(options), benchmark_repetitions, benchmark_max_tokens

    @staticmethod
    def _validate_integer(value, minimum, maximum):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Invalid request.")
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise ValueError("Invalid request.")
        if not minimum <= value <= maximum:
            raise ValueError("Invalid request.")
        return int(value)

    def _safe_callback_result(self, conversation, status):
        return (
            self._render_conversation(conversation),
            no_update,
            no_update,
            self._response_figure(conversation),
            status,
        )

    @staticmethod
    def _research_feedback_outcome(prompt):
        text = prompt.lower()
        if re.search(
            r"\b(?:that|this|your|the)\s+"
            r"(?:answers?|responses?|results?|conclusions?)"
            r"\s+(?:is|are|was|were|seems|looks)?\s*"
            r"(?:wrong|incorrect|false|inaccurate)\b"
            r"|\b(?:wrong|incorrect|false|inaccurate)\s+"
            r"(?:answers?|responses?|results?|conclusions?)\b"
            r"|\b(?:that|this)\s+(?:is|was)\s+"
            r"(?:wrong|incorrect|false|inaccurate)\b"
            r"|\b(?:answers?|responses?|results?|conclusions?)\b.{0,30}"
            r"\b(?:is|are|was|were)\s+not\s+(?:correct|accurate|true)\b",
            text,
        ):
            return "incorrect"
        if re.search(
            r"\b(?:that|this|your|the)\s+"
            r"(?:answers?|responses?|results?|conclusions?)"
            r"\s+(?:is|are|was|were)?\s*(?:correct|accurate|right|verified)\b"
            r"|\b(?:that|this)\s+(?:is|was)\s+"
            r"(?:correct|right|accurate)\b",
            text,
        ):
            return "correct"
        return None

    def _record_source_findings(self, evidence):
        try:
            inserted = self.history.record_source_findings(
                evidence,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        except Exception:
            self.logger.exception("Failed to store sourced research findings")
            return
        if inserted:
            self.logger.info("Saved %d new sourced research findings", inserted)

    @staticmethod
    def _is_research_review_request(prompt):
        return bool(
            re.search(
                r"\b(?:review|check|fact[\s-]?check|validate|verify)\b"
                r".{0,80}\b(?:results?|answer|response|claims?|query|sources?)\b"
                r"|\b(?:results?|answer|response|claims?)\b"
                r".{0,80}\b(?:review|check|fact[\s-]?check|validate|verify)\b",
                prompt,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _previous_research_topic(cls, history):
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            content = item.get("content", "")
            if (
                not cls._research_feedback_outcome(content)
                and not cls._is_research_review_request(content)
                and MultiAgentWorkflow._should_search_prompt(content)
            ):
                return content
        return ""

    def _is_sleeping(self):
        with self._sleep_state_lock:
            return self._sleep_pending or self._sleeping

    def _sleep_ui_state(self):
        with self._sleep_state_lock:
            sleep_pending = self._sleep_pending
            sleeping = self._sleeping
            sleep_status = self._sleep_status
        disabled = sleep_pending or sleeping or self._request_lock.locked()
        notice = (
            html.P(sleep_status, className="sleep-notice")
            if sleep_pending or sleeping
            else None
        )
        return disabled, disabled, notice

    def _start_sleep(self, reason):
        with self._sleep_state_lock:
            if self._sleep_pending or self._sleeping:
                return False
            self._sleep_pending = True
            self._sleep_status = (
                "The agent is ready to sleep and waiting for the current workflow "
                "to finish. Send and microphone are disabled; choose Wake to cancel."
            )
            self._sleep_wake_event.clear()
            self._sleep_thread = threading.Thread(
                target=self._run_sleep_cycle,
                args=(reason,),
                name="research-memory-sleep",
                daemon=True,
            )
            self._sleep_thread.start()
        self.logger.info("Research-memory sleep requested (%s)", reason)
        return True

    def _wake_sleep(self):
        with self._sleep_state_lock:
            if not self._sleep_pending and not self._sleeping:
                return False
            self._sleep_status = (
                "Canceling the pending sleep."
                if self._sleep_pending
                else "The agent is waking and will finish its current source check."
            )
            self._sleep_wake_event.set()
        self.logger.info("Research-memory sleep wake requested")
        return True

    def _run_sleep_cycle(self, reason):
        checked = 0
        updated = 0
        failed = 0
        acquired = False
        try:
            while not self._sleep_wake_event.is_set():
                if self._request_lock.acquire(timeout=0.5):
                    acquired = True
                    break
            if not acquired or self._sleep_wake_event.is_set():
                return

            with self._sleep_state_lock:
                self._sleep_pending = False
                self._sleeping = True
                self._sleep_status = (
                    "The agent is sleeping and reviewing older sourced memories. "
                    "Send and microphone are disabled; choose Wake to stop."
                )
            self.logger.info("Research-memory sleep started (%s)", reason)
            findings = self.history.get_source_findings_for_sleep(
                older_than_days=SLEEP_SOURCE_AGE_DAYS,
                limit=SLEEP_MAX_FINDINGS_PER_CYCLE,
            )
            total = len(findings)
            for index, finding in enumerate(findings, start=1):
                if self._sleep_wake_event.is_set():
                    break
                try:
                    fetched_source, fetch_error = self.workflow.agent._fetch_public_page(
                        finding["source_url"]
                    )
                    if fetched_source is None:
                        failed += 1
                        self.logger.warning(
                            "Sleep could not refresh a source finding (%s)",
                            fetch_error,
                        )
                    else:
                        changed = self.history.refresh_source_finding(
                            finding,
                            fetched_source,
                            datetime.now().astimezone().isoformat(timespec="seconds"),
                        )
                        checked += 1
                        updated += int(changed)
                    with self._sleep_state_lock:
                        self._sleep_status = (
                            "The agent is sleeping and integrating sourced memories "
                            f"({index}/{total} checked, {updated} updated, {failed} unavailable). "
                            "Send and microphone are disabled; choose Wake to stop."
                        )
                except Exception:
                    failed += 1
                    self.logger.exception(
                        "Sleep failed while refreshing a source finding"
                    )
                if self._sleep_wake_event.wait(0.25):
                    break
        except Exception:
            self.logger.exception("Research-memory sleep cycle failed")
            with self._sleep_state_lock:
                self._sleep_status = "The agent's sleep cycle ended with an error."
        finally:
            self.logger.info(
                "Research-memory sleep completed (%d checked, %d updated, %d unavailable)",
                checked,
                updated,
                failed,
                extra={"reason": reason},
            )
            if acquired:
                self._request_lock.release()
            with self._sleep_state_lock:
                self._sleep_pending = False
                self._sleeping = False
                self._sleep_status = ""
                self._sleep_thread = None

    @staticmethod
    def _build_word_document(content):
        """Create a styled Word document from the latest Markdown response."""
        document = Document()
        document.core_properties.title = "Assistant response"
        document.core_properties.subject = "Latest assistant response"
        section = document.sections[0]
        section.top_margin = section.bottom_margin = Inches(0.7)
        section.left_margin = section.right_margin = Inches(0.8)
        document.add_heading("Assistant response", level=0)

        def add_text_runs(paragraph, text, code=False):
            if code:
                run = paragraph.add_run(text)
                run.font.name = "Consolas"
                return
            tokens = re.split(r"(\*\*.+?\*\*|__.+?__|`.+?`|\*.+?\*)", text)
            for token in tokens:
                if not token:
                    continue
                if token.startswith("**") and token.endswith("**"):
                    paragraph.add_run(token[2:-2]).bold = True
                elif token.startswith("__") and token.endswith("__"):
                    paragraph.add_run(token[2:-2]).bold = True
                elif token.startswith("`") and token.endswith("`"):
                    run = paragraph.add_run(token[1:-1])
                    run.font.name = "Consolas"
                elif token.startswith("*") and token.endswith("*"):
                    paragraph.add_run(token[1:-1]).italic = True
                else:
                    paragraph.add_run(token)

        lines = content.splitlines()
        index = 0
        in_code_block = False
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                index += 1
                continue
            if in_code_block:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_after = 0
                add_text_runs(paragraph, lines[index], code=True)
            elif not stripped:
                index += 1
                continue
            elif stripped.startswith("|") and stripped.endswith("|"):
                table_lines = []
                while (
                    index < len(lines)
                    and lines[index].strip().startswith("|")
                    and lines[index].strip().endswith("|")
                ):
                    table_lines.append(lines[index].strip())
                    index += 1
                rows = [
                    [cell.strip() for cell in row.strip("|").split("|")]
                    for row in table_lines
                ]
                if len(rows) > 1 and all(
                    re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                    for cell in rows[1]
                ):
                    rows.pop(1)
                column_count = max((len(row) for row in rows), default=0)
                if column_count:
                    table = document.add_table(rows=1, cols=column_count)
                    table.style = "Light Shading Accent 1"
                    for row_index, cells in enumerate(rows):
                        target_cells = (
                            table.rows[0].cells
                            if row_index == 0
                            else table.add_row().cells
                        )
                        for cell_index, cell_text in enumerate(cells):
                            if cell_index < column_count:
                                target_cell = target_cells[cell_index]
                                target_cell.text = ""
                                add_text_runs(
                                    target_cell.paragraphs[0],
                                    cell_text,
                                )
                continue
            elif re.match(r"^#{1,6}\s+", stripped):
                heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
                paragraph = document.add_heading(
                    level=min(len(heading.group(1)), 4)
                )
                add_text_runs(paragraph, heading.group(2))
            elif re.match(r"^[-*+]\s+", stripped):
                paragraph = document.add_paragraph(style="List Bullet")
                add_text_runs(paragraph, re.sub(r"^[-*+]\s+", "", stripped))
            elif re.match(r"^\d+[.)]\s+", stripped):
                paragraph = document.add_paragraph(style="List Number")
                add_text_runs(paragraph, re.sub(r"^\d+[.)]\s+", "", stripped))
            else:
                paragraph = document.add_paragraph()
                add_text_runs(paragraph, stripped)
            index += 1

        output = BytesIO()
        document.save(output)
        return output.getvalue()

    def _start_sleep_scheduler(self):
        with self._sleep_state_lock:
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                return
            self._scheduler_stop_event.clear()
            self._scheduler_thread = threading.Thread(
                target=self._run_sleep_scheduler,
                name="research-memory-sleep-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        self.logger.info("Research-memory sleep scheduled every 24 hours")

    def _run_sleep_scheduler(self):
        while not self._scheduler_stop_event.wait(
            SLEEP_SCHEDULE_INTERVAL_SECONDS
        ):
            self._start_sleep("scheduled")

    @staticmethod
    def _render_research_memory(memories):
        if not memories:
            return html.P(
                "No ranked research memories have been recorded.",
                className="empty-state",
            )
        rows = [
            html.Tr(
                [
                    html.Td(memory["category"]),
                    html.Td(memory["topic"]),
                    html.Td(f"{memory['rank']:+d}"),
                    html.Td(
                        f"{memory['correct_count']} correct / "
                        f"{memory['incorrect_count']} incorrect / "
                        f"{memory['review_count']} reviewed"
                    ),
                    html.Td(
                        html.Ul(
                            [
                                html.Li(
                                    f"{event['outcome']}: {event['feedback']}"
                                )
                                for event in memory["events"]
                            ]
                        )
                    ),
                    html.Td(memory["lesson"]),
                ]
            )
            for memory in memories
        ]
        return html.Div(
            [
                html.H2("Ranked research memory"),
                html.P(
                    "Ranks reflect explicit user feedback, not verified truth. "
                    "Memory is never a substitute for current source evidence."
                ),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Category"),
                                    html.Th("Topic"),
                                    html.Th("Rank"),
                                    html.Th("Feedback"),
                                    html.Th("Recent feedback events"),
                                    html.Th("Lesson"),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="research-memory-table",
                ),
            ]
        )

    @staticmethod
    def _get_conversation_id():
        conversation_id = session.get("conversation_id")
        if not isinstance(conversation_id, str) or len(conversation_id) != 32:
            conversation_id = uuid.uuid4().hex
            session["conversation_id"] = conversation_id
        session.permanent = True
        return conversation_id

    def run(self, host="127.0.0.1", port=8050):
        """Start the Flask development server hosting the Dash application."""
        self._start_sleep_scheduler()
        self.server.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    AIServer().run()
