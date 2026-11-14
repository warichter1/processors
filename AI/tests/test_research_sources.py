import sqlite3
import threading
import unittest
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from AI.AIServer import AIServer
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
