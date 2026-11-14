import sqlite3
import threading
import unittest
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from AI.AIServer import AIServer, MENU_WORKFLOW_MODES
from AI.conversation_history import ConversationHistory
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from AI.multiAgent import (
    CHAT_ARCHITECT_TASK_PROMPT,
    CHAT_FACTUAL_EXPECTED_OUTPUT,
    MultiAgent,
    MultiAgentWorkflow,
    _DuckDuckGoResultParser,
)


class ResearchSourceTests(unittest.TestCase):
    def test_chat_prompt_requires_row_citations_and_separates_recommendations(self):
        self.assertIn("cite each factual row", CHAT_ARCHITECT_TASK_PROMPT)
        self.assertIn("related product families separate", CHAT_ARCHITECT_TASK_PROMPT)
        self.assertIn("label recommendations as recommendations", CHAT_ARCHITECT_TASK_PROMPT)
        self.assertIn("cite each table row", CHAT_FACTUAL_EXPECTED_OUTPUT)

    def test_researched_table_requires_valid_citation_in_every_data_row(self):
        evidence = "[S1] Official release\nURL: https://example.com"
        answer = (
            "| Model | Parameters | Sources |\n"
            "|---|---:|---|\n"
            "| Model A | 9B | [S1] |\n"
            "| Model B | 27B | [S2] |"
        )

        issues = MultiAgentWorkflow._table_citation_issues(answer, evidence)

        self.assertEqual(len(issues), 2)
        self.assertIn("unknown source IDs", issues[0])
        self.assertIn("table row 4", issues[1])

    def test_researched_table_rejects_rows_without_citations(self):
        evidence = "[S1] Official release\nURL: https://example.com"
        answer = (
            "| Model | Parameters |\n"
            "|---|---:|\n"
            "| Model A | 9B [S1] |\n"
            "| Model B | 27B |"
        )

        issues = MultiAgentWorkflow._table_citation_issues(answer, evidence)

        self.assertEqual(len(issues), 1)
        self.assertIn("table row 4", issues[0])

    def test_researched_table_accepts_citations_from_supplied_sources(self):
        evidence = (
            "[S1] Official release\nURL: https://example.com\n\n"
            "[S2] Official collection\nURL: https://example.org"
        )
        answer = (
            "| Model | Parameters | Sources |\n"
            "|---|---:|---|\n"
            "| Model A | 9B | [S1] |\n"
            "| Model B | 27B | [S1], [S2] |"
        )

        self.assertEqual(
            MultiAgentWorkflow._table_citation_issues(answer, evidence),
            [],
        )

    def test_researched_table_requires_source_evidence(self):
        answer = "| Model | Parameters |\n|---|---:|\n| Model A | 9B [S1] |"

        self.assertTrue(
            MultiAgentWorkflow._table_citation_issues(answer, "")
        )

    def test_search_snippets_remain_associated_with_their_result(self):
        parser = _DuckDuckGoResultParser()
        parser.feed(
            '<a class="result__a" href="https://example.com/one">One</a>'
            '<a class="result__a" href="https://example.com/two">Two</a>'
            '<div class="result__snippet">Evidence from result two.</div>'
        )

        self.assertIsNone(parser.results[0]["snippet"])
        self.assertEqual(
            parser.results[1]["snippet"],
            "Evidence from result two.",
        )

    def test_page_extraction_ignores_navigation_and_script_text(self):
        title, text = MultiAgent._extract_page_text(
            "<html><head><title>Official Release</title>"
            "<script>fake version 99</script></head>"
            "<body><nav>Navigation noise</nav>"
            "<main><h1>Version 3.8</h1><p>Official release details.</p></main>"
            "</body></html>"
        )

        self.assertEqual(title, "Official Release")
        self.assertEqual(text, "Version 3.8 Official release details.")

    @patch("AI.multiAgent.socket.getaddrinfo")
    def test_public_url_validation_rejects_private_addresses_and_tokens(
            self, getaddrinfo
    ):
        self.assertIsNone(MultiAgent._validated_public_url("http://127.0.0.1/admin"))
        self.assertIsNone(
            MultiAgent._validated_public_url(
                "https://example.com/page?access_token=secret"
            )
        )

        getaddrinfo.return_value = [
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ]
        self.assertEqual(
            MultiAgent._validated_public_url("https://example.com/page"),
            "https://example.com/page",
        )

    @patch("AI.multiAgent.socket.getaddrinfo")
    @patch("AI.multiAgent.requests.get")
    def test_fetch_rejects_redirects_to_private_networks(
            self, request_get, getaddrinfo
    ):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ]
        redirect = Mock()
        redirect.status_code = 302
        redirect.headers = {"Location": "http://127.0.0.1/private"}
        request_get.return_value = redirect
        agent = MultiAgent.__new__(MultiAgent)

        page, error = agent._fetch_public_page("https://example.com/start")

        self.assertIsNone(page)
        self.assertEqual(error, "unsafe_or_invalid_url")
        redirect.close.assert_called_once()
        request_get.assert_called_once()

    @patch("AI.multiAgent.socket.getaddrinfo")
    @patch("AI.multiAgent.requests.get")
    def test_fetch_extracts_bounded_html_page_text(self, request_get, getaddrinfo):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ]
        response = Mock()
        response.status_code = 200
        response.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": "256",
        }
        response.encoding = "utf-8"
        response.iter_content.return_value = [
            b"<title>Release</title><nav>noise</nav>"
            b"<article><p>Version 3.8 is documented.</p></article>"
        ]
        request_get.return_value = response
        agent = MultiAgent.__new__(MultiAgent)

        page, error = agent._fetch_public_page("https://example.com/release")

        self.assertIsNone(error)
        self.assertEqual(page["title"], "Release")
        self.assertEqual(page["text"], "Version 3.8 is documented.")
        response.close.assert_called_once()

    def test_search_query_removes_url_query_and_fragment(self):
        query = MultiAgent._sanitize_search_query(
            "Summarize https://example.com/path?token=secret#section"
        )

        self.assertEqual(query, "Summarize https://example.com/path")
        self.assertNotIn("secret", query)

    @patch("AI.multiAgent.socket.getaddrinfo")
    @patch("AI.multiAgent.requests.get")
    def test_search_output_cites_fetched_page_with_stable_source_id(
            self, request_get, getaddrinfo
    ):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ]
        search_response = Mock()
        search_response.text = (
            '<a class="result__a" href="https://example.com/release">'
            "Official release</a>"
            '<div class="result__snippet">Search summary</div>'
        )
        search_response.raise_for_status = Mock()
        page_response = Mock()
        page_response.status_code = 200
        page_response.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": "256",
        }
        page_response.encoding = "utf-8"
        page_response.iter_content.return_value = [
            b"<title>Release page</title><article>"
            b"<p>Version 3.8 was released.</p></article>"
        ]
        request_get.side_effect = [search_response, page_response]
        agent = MultiAgent.__new__(MultiAgent)
        agent._log_profile = Mock()

        evidence = agent._create_internet_search_tool().run(
            query="Read https://example.com/release",
            max_results=5,
        )

        self.assertIn("[S1] Official release", evidence)
        self.assertIn("URL: https://example.com/release", evidence)
        self.assertIn("Fetched page excerpt: Version 3.8 was released.", evidence)
        self.assertEqual(
            request_get.call_args_list[0].kwargs["params"]["q"],
            "Read https://example.com/release",
        )


class LatencyHistoryTests(unittest.TestCase):
    def test_existing_latency_schema_migrates_without_changing_old_fields(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE conversation_latency (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        duration_s REAL NOT NULL,
                        context_messages INTEGER NOT NULL,
                        context_chars INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO conversation_latency
                        (session_id, duration_s, context_messages, context_chars, created_at)
                    VALUES ('old-session', 12.5, 3, 500, '2026-11-12T00:00:00+00:00')
                    """
                )

            history = ConversationHistory(str(database))
            history.append_turn(
                "new-session",
                "question",
                "answer",
                1.5,
                3,
                550,
                "2026-11-12T00:01:00+00:00",
                history_context_messages=2,
                history_context_chars=500,
                prompt_chars=50,
            )

            with sqlite3.connect(database) as connection:
                old_row = connection.execute(
                    """
                    SELECT context_messages, context_chars,
                           history_context_messages, history_context_chars, prompt_chars
                    FROM conversation_latency
                    WHERE session_id = 'old-session'
                    """
                ).fetchone()
                new_row = connection.execute(
                    """
                    SELECT context_messages, context_chars,
                           history_context_messages, history_context_chars, prompt_chars
                    FROM conversation_latency
                    WHERE session_id = 'new-session'
                    """
                ).fetchone()

            self.assertEqual(old_row, (3, 500, 0, 0, 0))
            self.assertEqual(new_row, (3, 550, 2, 500, 50))


class FactualResearchMemoryTests(unittest.TestCase):
    @staticmethod
    def _component_text(component):
        if isinstance(component, str):
            return component
        children = getattr(component, "children", None)
        if isinstance(children, list):
            return " ".join(
                FactualResearchMemoryTests._component_text(child)
                for child in children
            )
        return FactualResearchMemoryTests._component_text(children) if children else ""

    @staticmethod
    def _component_count(component, class_name):
        if isinstance(component, list):
            return sum(
                FactualResearchMemoryTests._component_count(child, class_name)
                for child in component
            )
        if component is None or isinstance(component, str):
            return 0
        children = getattr(component, "children", None)
        return int(component.__class__.__name__ == class_name) + (
            FactualResearchMemoryTests._component_count(children, class_name)
        )

    def test_empty_ranked_memory_explains_explicit_feedback_process(self):
        rendered = AIServer._render_research_memory([])
        text = self._component_text(rendered)

        self.assertIn("No saved conversation turns", text)
        self.assertIn("unvalidated candidates", text)
        self.assertIn("rank 0", text)
        self.assertIn("correct", text)
        self.assertIn("incorrect", text)

    def test_dialog_renderers_format_workflow_output_and_nested_menu_guide(self):
        result = AIServer._render_workflow_result("Benchmark results", "# Completed")
        self.assertEqual(result.children[0].children, "Benchmark results")
        self.assertEqual(result.children[1].children, "# Completed")

        guide = AIServer._render_help_dialog()
        text = self._component_text(guide)
        self.assertIn("Generate Script", text)
        self.assertIn("Benchmark Models", text)
        self.assertIn("View Ranked Research Memory", text)
        self.assertIn("Chat window width selects a width", text)
        self.assertGreaterEqual(self._component_count(guide, "Ul"), 3)

    def test_ranked_memory_table_reuses_chat_result_table_style_and_markdown(self):
        memory_table = AIServer._render_research_memory(
            [
                {
                    "category": "software",
                    "topic": "Compare package options",
                    "rank": 2,
                    "correct_count": 1,
                    "incorrect_count": 0,
                    "review_count": 1,
                    "events": [
                        {
                            "outcome": "correct",
                            "feedback": (
                                "| Package | License |\n"
                                "| ------- | ------- |\n"
                                "| Alpha | MIT |"
                            ),
                        }
                    ],
                    "lesson": (
                        "| Package | Recommendation |\n"
                        "| ------- | -------------- |\n"
                        "| Alpha | Use for the example |"
                    ),
                }
            ]
        )
        wrapper = memory_table.children[2]
        table = wrapper.children
        row = table.children[1].children[0]

        self.assertIn("wide-result-table", table.className.split())
        event_markdown = row.children[4].children.children[0].children[1]
        lesson_markdown = row.children[5].children
        self.assertEqual(event_markdown.__class__.__name__, "Markdown")
        self.assertIn("| ------- |", event_markdown.children)
        self.assertEqual(lesson_markdown.__class__.__name__, "Markdown")
        self.assertIn("| -------------- |", lesson_markdown.children)

    def test_saved_turns_integrate_once_and_only_feedback_changes_persona_rank(self):
        with TemporaryDirectory() as directory:
            history = ConversationHistory(
                str(Path(directory) / "conversation-history.sqlite3")
            )
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            history.append_turn(
                "session",
                "Compare Qwen model context sizes",
                "Qwen context sizes vary by model; verify against current model cards.",
                1.0,
                1,
                40,
                created_at,
            )

            self.assertEqual(history.integrate_conversation_memories(), 1)
            self.assertEqual(history.integrate_conversation_memories(), 0)
            [candidate] = history.get_research_memory()
            self.assertEqual(candidate["rank"], 0)
            self.assertEqual(candidate["review_count"], 1)
            self.assertEqual(
                history.recall_research_memory(
                    "session",
                    "Qwen model context sizes",
                    created_at,
                ),
                [],
            )

            history.record_research_memory(
                "session",
                "Compare Qwen model context sizes",
                "Correction: distinguish the model variants and cite their model cards.",
                "incorrect",
                created_at,
            )
            self.assertEqual(history.rank_research_memory(), 1)
            [learned] = history.get_research_memory()
            self.assertEqual(learned["rank"], -2)
            recalled = history.recall_research_memory(
                "session",
                "Qwen model context sizes",
                created_at,
            )
            self.assertEqual(len(recalled), 1)
            self.assertIn("distinguish the model variants", recalled[0]["lesson"])

    def test_feedback_parser_recognizes_natural_positive_and_negative_feedback(self):
        self.assertEqual(AIServer._research_feedback_outcome("Correct."), "correct")
        self.assertEqual(
            AIServer._research_feedback_outcome("That was right, thanks."),
            "correct",
        )
        self.assertEqual(AIServer._research_feedback_outcome("Wrong."), "incorrect")
        self.assertEqual(
            AIServer._research_feedback_outcome(
                "You're wrong; Qwen 3 and Qwen 3.5 have different context lengths."
            ),
            "incorrect",
        )

    def test_sleep_settings_have_requested_defaults_and_enforce_bounds(self):
        self.assertEqual(AIServer._validated_sleep_time("04:00"), "04:00")
        self.assertEqual(AIServer._validated_sleep_cycle_minutes(30), 30)
        self.assertEqual(
            AIServer._seconds_until_next_sleep(
                "04:00",
                datetime.fromisoformat("2026-11-14T03:00:00-05:00"),
            ),
            3600,
        )
        for invalid in ("25:00", "not a time", None):
            with self.subTest(schedule_time=invalid):
                with self.assertRaises(ValueError):
                    AIServer._validated_sleep_time(invalid)
        for invalid in (0, 1441, 1.5, True, "bad"):
            with self.subTest(max_cycle=invalid):
                with self.assertRaises(ValueError):
                    AIServer._validated_sleep_cycle_minutes(invalid)

    def test_menu_actions_route_directly_to_workflow_callback(self):
        self.assertEqual(MENU_WORKFLOW_MODES["menu-view-memory"], "view-memory")
        self.assertEqual(
            MENU_WORKFLOW_MODES["menu-generate-script"],
            "generate",
        )
        self.assertEqual(
            MENU_WORKFLOW_MODES["menu-benchmark-models"],
            "benchmark",
        )

        with TemporaryDirectory() as directory:
            server = AIServer(
                target_dir=directory,
                debug=False,
                web_search=False,
            )
            self.assertEqual(server.sleep_schedule_time, "04:00")
            self.assertEqual(server.sleep_max_cycle_minutes, 30)
            self.assertEqual(
                server.history.get_config("sleep_schedule_time"),
                "04:00",
            )
        workflow_callback = next(
            callback
            for callback in server.dash_app.callback_map.values()
            if any(
                output.component_id == "conversation-view"
                for output in (
                    callback["output"]
                    if isinstance(callback["output"], (list, tuple))
                    else [callback["output"]]
                )
            )
        )
        input_ids = {item["id"] for item in workflow_callback["inputs"]}
        self.assertTrue(
            {
                "menu-view-memory",
                "menu-generate-script",
                "menu-benchmark-models",
            }.issubset(input_ids)
        )
        dialog_callback = next(
            callback
            for callback in server.dash_app.callback_map.values()
            if any(
                output.component_id == "help-modal"
                for output in (
                    callback["output"]
                    if isinstance(callback["output"], (list, tuple))
                    else [callback["output"]]
                )
            )
        )
        self.assertIn(
            "request-status",
            {item["id"] for item in dialog_callback["inputs"]},
        )
        self.assertIn(
            "menu-sleep-schedule",
            {item["id"] for item in dialog_callback["inputs"]},
        )
        self.assertIn(
            "menu-sleep-settings-save",
            {item["id"] for item in dialog_callback["inputs"]},
        )

    def test_health_endpoint_exposes_uncached_server_instance_id(self):
        with TemporaryDirectory() as directory:
            server = AIServer(
                target_dir=directory,
                debug=False,
                web_search=False,
            )
            client = server.server.test_client()

            first_response = client.get("/healthz")
            second_response = client.get("/healthz")

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(first_response.get_json()["status"], "ok")
        self.assertEqual(
            first_response.get_json()["instance_id"],
            second_response.get_json()["instance_id"],
        )
        self.assertEqual(first_response.headers["Cache-Control"], "no-store")

    def test_chat_width_is_validated_and_applied_as_max_width(self):
        self.assertEqual(
            AIServer._chat_width_style(2000),
            {"maxWidth": "2000px"},
        )
        for invalid_width in (True, 1199, 1300, 2401, "1600"):
            with self.subTest(width=invalid_width):
                with self.assertRaises(ValueError):
                    AIServer._chat_width_style(invalid_width)

    def test_about_dialog_shows_class_version_and_server_uptime(self):
        server = object.__new__(AIServer)
        server.started_at = datetime(2026, 11, 14, 14, 0).astimezone()
        server._started_monotonic = 100.0
        with patch("AI.AIServer.perf_counter", return_value=3_800.0):
            dialog = server._render_about_dialog()

        self.assertEqual(
            dialog.children[1].children.children,
            "Multi-Agent Assistant",
        )
        self.assertEqual(
            dialog.children[2].children[0].children,
            "Version: ",
        )
        self.assertEqual(
            dialog.children[2].children[1],
            AIServer.VERSION,
        )
        self.assertEqual(
            dialog.children[3].children[0].children,
            "Web server started: ",
        )
        self.assertEqual(
            dialog.children[4].children[0].children,
            "Uptime: ",
        )
        self.assertEqual(dialog.children[0].children, "About")
        self.assertEqual(dialog.className, "dialog-section")
        self.assertEqual(
            dialog.children[4].children[1],
            "1h 1m 40s",
        )

    def test_uptime_format_includes_days_and_zero_seconds(self):
        self.assertEqual(AIServer._format_uptime(172_800), "2d 0h 0m 0s")

    def test_page_guide_has_formatted_chat_and_menu_sections(self):
        guide = AIServer._render_help_dialog()

        self.assertEqual(guide.children[0].children, "Page guide")
        sections = guide.children[1:]
        self.assertEqual(
            [section.children[0].children for section in sections],
            ["Chat page", "Menus"],
        )
        self.assertEqual(len(sections[0].children[1].children), 6)
        self.assertEqual(len(sections[1].children[1].children), 6)

    def test_sleep_state_migrates_existing_source_memory_table(self):
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "conversation-history.sqlite3"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE research_facts (
                        fact_key TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        source_title TEXT NOT NULL,
                        excerpt TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        verified_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO research_facts
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy-key",
                        "Qwen",
                        "Model card",
                        "Qwen details.",
                        "https://example.com/qwen",
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ),
                )

            history = ConversationHistory(str(database_path))

            with history._connect() as connection:
                row = connection.execute(
                    "SELECT status FROM research_facts WHERE fact_key = ?",
                    ("legacy-key",),
                ).fetchone()
            self.assertEqual(row["status"], "active")

    def test_only_new_fetched_source_excerpts_are_saved_and_recalled(self):
        with TemporaryDirectory() as directory:
            history = ConversationHistory(
                str(Path(directory) / "conversation-history.sqlite3")
            )
            evidence = (
                "[S1] Official Qwen model card\n"
                "URL: https://example.com/qwen\n"
                "Fetched page excerpt: Qwen 3.5 9B has a 262144 token context.\n\n"
                "[S2] Search result\n"
                "URL: https://example.org/qwen\n"
                "Search snippet: Unverified model claims."
            )
            verified_at = datetime.now().astimezone().isoformat(timespec="seconds")

            self.assertEqual(
                history.record_source_findings(evidence, verified_at),
                1,
            )
            self.assertEqual(
                history.record_source_findings(evidence, verified_at),
                0,
            )

            recalled = history.recall_source_findings("Qwen 3.5 model context")

            self.assertEqual(len(recalled), 1)
            self.assertIn("262144 token context", recalled[0]["excerpt"])
            self.assertEqual(
                recalled[0]["source_url"],
                "https://example.com/qwen",
            )

    def test_stale_source_findings_are_not_recalled(self):
        with TemporaryDirectory() as directory:
            history = ConversationHistory(
                str(Path(directory) / "conversation-history.sqlite3")
            )
            evidence = (
                "[S1] Official Qwen model card\n"
                "URL: https://example.com/qwen\n"
                "Fetched page excerpt: Qwen 3.5 model details."
            )
            verified_at = (
                    datetime.now().astimezone() - timedelta(days=31)
            ).isoformat(timespec="seconds")
            history.record_source_findings(evidence, verified_at)

            self.assertEqual(
                history.recall_source_findings("Qwen 3.5 model"),
                [],
            )

    def test_refetching_an_existing_finding_refreshes_verification_without_duplicate(self):
        with TemporaryDirectory() as directory:
            history = ConversationHistory(
                str(Path(directory) / "conversation-history.sqlite3")
            )
            evidence = (
                "[S1] Official Qwen model card\n"
                "URL: https://example.com/qwen\n"
                "Fetched page excerpt: Qwen 3.5 model details."
            )
            old_timestamp = (
                    datetime.now().astimezone() - timedelta(days=31)
            ).isoformat(timespec="seconds")
            new_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            history.record_source_findings(evidence, old_timestamp)

            self.assertEqual(
                history.record_source_findings(evidence, new_timestamp),
                0,
            )
            self.assertEqual(
                len(history.recall_source_findings("Qwen 3.5 model")),
                1,
            )

    def test_sleep_replaces_changed_excerpts_and_only_recalls_current_version(self):
        with TemporaryDirectory() as directory:
            history = ConversationHistory(
                str(Path(directory) / "conversation-history.sqlite3")
            )
            finding_text = (
                "[S1] Official Qwen model card\n"
                "URL: https://example.com/qwen\n"
                "Fetched page excerpt: Qwen 3.5 has a 128K context."
            )
            old_timestamp = (
                    datetime.now().astimezone() - timedelta(days=8)
            ).isoformat(timespec="seconds")
            history.record_source_findings(finding_text, old_timestamp)
            [finding] = history.get_source_findings_for_sleep()

            updated = history.refresh_source_finding(
                finding,
                {
                    "title": "Updated Qwen model card",
                    "text": "Qwen 3.5 has a 256K context.",
                },
                datetime.now().astimezone().isoformat(timespec="seconds"),
            )

            self.assertTrue(updated)
            recalled = history.recall_source_findings("Qwen 3.5 context")
            self.assertEqual(len(recalled), 1)
            self.assertIn("256K context", recalled[0]["excerpt"])
            self.assertEqual(
                history.get_source_findings_for_sleep(),
                [],
            )
            with history._connect() as connection:
                statuses = [
                    row[0]
                    for row in connection.execute(
                        "SELECT status FROM research_facts ORDER BY verified_at"
                    )
                ]
            self.assertEqual(statuses, ["superseded", "active"])

    def test_sleep_cycle_refreshes_sources_and_exits_awake(self):
        finding = {
            "fact_key": "existing-key",
            "topic": "Qwen 3.5 model",
            "source_title": "Qwen model card",
            "excerpt": "Older excerpt.",
            "source_url": "https://example.com/qwen",
            "verified_at": "2026-01-01T00:00:00+00:00",
        }
        server = object.__new__(AIServer)
        server._sleep_state_lock = threading.Lock()
        server._sleep_wake_event = threading.Event()
        server._request_lock = threading.Lock()
        server._sleep_pending = False
        server._sleeping = True
        server._sleep_status = "Sleeping."
        server._sleep_thread = object()
        server.history = Mock()
        server.history.get_source_findings_for_sleep.return_value = [finding]
        server.history.refresh_source_finding.return_value = True
        server.workflow = Mock()
        server.workflow.agent._fetch_public_page.return_value = (
            {"title": "Qwen model card", "text": "New excerpt."},
            None,
        )
        server.logger = Mock()

        server._run_sleep_cycle("test")

        server.history.refresh_source_finding.assert_called_once()
        self.assertFalse(server._sleeping)
        self.assertIsNone(server._sleep_thread)
        self.assertEqual(server._sleep_status, "")
        (
            send_disabled,
            microphone_disabled,
            notice,
            polling_disabled,
        ) = server._sleep_ui_state()
        self.assertFalse(send_disabled)
        self.assertFalse(microphone_disabled)
        self.assertIsNone(notice)
        self.assertTrue(polling_disabled)

    def test_sleeping_ui_disables_buttons_and_displays_notice(self):
        server = object.__new__(AIServer)
        server._sleep_state_lock = threading.Lock()
        server._request_lock = threading.Lock()
        server._sleep_pending = False
        server._sleeping = True
        server._sleep_status = "The agent is sleeping."

        (
            send_disabled,
            microphone_disabled,
            notice,
            polling_disabled,
        ) = server._sleep_ui_state()

        self.assertTrue(send_disabled)
        self.assertTrue(microphone_disabled)
        self.assertEqual(notice.children, "The agent is sleeping.")
        self.assertFalse(polling_disabled)

    def test_starting_nap_disables_controls_and_displays_start_notice(self):
        server = object.__new__(AIServer)
        server._sleep_state_lock = threading.Lock()
        server._sleep_wake_event = threading.Event()
        server._request_lock = threading.Lock()
        server._sleep_pending = False
        server._sleeping = False
        server._sleep_status = ""
        server._sleep_thread = None
        server.logger = Mock()

        with patch("AI.AIServer.threading.Thread") as thread_class:
            self.assertTrue(server._start_sleep("test"))
        thread_class.return_value.start.assert_called_once()
        (
            send_disabled,
            microphone_disabled,
            notice,
            polling_disabled,
        ) = server._sleep_ui_state()

        self.assertTrue(send_disabled)
        self.assertTrue(microphone_disabled)
        self.assertEqual(notice.children, "The agent is starting a nap.")
        self.assertFalse(polling_disabled)

    def test_idle_sleep_ui_disables_polling(self):
        server = object.__new__(AIServer)
        server._sleep_state_lock = threading.Lock()
        server._request_lock = threading.Lock()
        server._sleep_pending = False
        server._sleeping = False
        server._sleep_status = ""

        send_disabled, microphone_disabled, notice, polling_disabled = (
            server._sleep_ui_state()
        )

        self.assertFalse(send_disabled)
        self.assertFalse(microphone_disabled)
        self.assertIsNone(notice)
        self.assertTrue(polling_disabled)

    def test_sleep_waits_for_active_workflow_and_shows_ready_notice(self):
        server = object.__new__(AIServer)
        server._sleep_state_lock = threading.Lock()
        server._sleep_wake_event = threading.Event()
        server._request_lock = threading.Lock()
        server._request_lock.acquire()
        server._sleep_pending = True
        server._sleeping = False
        server._sleep_status = "Ready to sleep; waiting for the current workflow."
        server._sleep_thread = None
        server.history = Mock()
        server.history.get_source_findings_for_sleep.return_value = []
        server.history.integrate_conversation_memories.return_value = 3
        server.history.rank_research_memory.return_value = 2
        server.history.integrate_conversation_memories.return_value = 0
        server.history.rank_research_memory.return_value = 0
        server.logger = Mock()

        sleep_thread = threading.Thread(
            target=server._run_sleep_cycle,
            args=("test",),
        )
        sleep_thread.start()
        (
            send_disabled,
            microphone_disabled,
            notice,
            polling_disabled,
        ) = server._sleep_ui_state()

        self.assertTrue(send_disabled)
        self.assertTrue(microphone_disabled)
        self.assertIn("Ready to sleep", notice.children)
        self.assertFalse(polling_disabled)
        server.history.get_source_findings_for_sleep.assert_not_called()

        server._request_lock.release()
        sleep_thread.join(timeout=2)

        self.assertFalse(sleep_thread.is_alive())
        server.history.get_source_findings_for_sleep.assert_called_once()
        server.history.integrate_conversation_memories.assert_called_once()
        server.history.rank_research_memory.assert_called_once()
        self.assertFalse(server._sleep_pending)
        self.assertFalse(server._sleeping)
        self.assertFalse(server._request_lock.locked())

    def test_word_export_formats_markdown_headings_lists_and_tables(self):
        content = (
            "# Results\n"
            "A **bold** claim and `inline code`.\n\n"
            "- First item\n\n"
            "| Model | Parameters |\n"
            "|---|---:|\n"
            "| Qwen | 9B |"
        )

        document = Document(BytesIO(AIServer._build_word_document(content)))

        self.assertEqual(document.paragraphs[0].text, "Assistant response")
        self.assertEqual(document.paragraphs[1].text, "Results")
        self.assertEqual(document.paragraphs[1].style.name, "Heading 1")
        self.assertEqual(document.paragraphs[2].text, "A bold claim and inline code.")
        self.assertTrue(any(run.bold for run in document.paragraphs[2].runs))
        self.assertEqual(document.paragraphs[3].style.name, "List Bullet")
        self.assertEqual(len(document.tables), 1)
        self.assertEqual(document.tables[0].rows[1].cells[0].text, "Qwen")
        self.assertEqual(document.tables[0].rows[1].cells[1].text, "9B")
        self.assertEqual(document.sections[0].orientation, WD_ORIENT.LANDSCAPE)
        for header_cell in document.tables[0].rows[0].cells:
            self.assertIsNotNone(header_cell._tc.tcPr.find(qn("w:noWrap")))
            self.assertTrue(all(run.bold for run in header_cell.paragraphs[0].runs))


if __name__ == "__main__":
    unittest.main()
