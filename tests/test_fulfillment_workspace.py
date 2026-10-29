"""User journeys for explicit decisions and honest fulfillment workspace scope."""

from contextlib import contextmanager
import copy
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from backend.fulfillment import plan_digest
from backend.fulfillment_state import current_result
from backend.project_io import loads_project


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
SCRIPT = '''
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_fulfillment import data_fulfillment
from frontend.pages.data_project import data_project

StateManager.init_session()
page = st.selectbox("Journey page", ["fulfillment", "project"], key="journey_page")
if st.button("Clear journey session", key="journey_clear"):
    StateManager.clear()
    st.rerun()
if page == "fulfillment":
    data_fulfillment()
else:
    data_project()
'''


class FulfillmentWorkspaceTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run(timeout=30)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def rendered_text(self, app):
        return "\n".join(str(item.value) for kind in (
            "caption", "info", "warning", "error", "markdown", "success", "subheader", "text",
        ) for item in getattr(app, kind))

    @contextmanager
    def uploads(self, **values):
        with patch("streamlit.file_uploader",
                   side_effect=lambda *args, **kwargs: values.get(kwargs.get("key"))):
            yield

    def load_bytes(self, app, data):
        with self.uploads(fulfillment_source_file=BytesIO(data)):
            app.run()
            app.button(key="fulfillment_load").click().run()
        self.assert_clean(app)

    def load(self, app, name):
        self.load_bytes(app, (FIXTURES / name).read_bytes())

    def run_plan(self, app):
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)

    def run_a(self, app):
        self.load(app, "month_a.csv")
        self.run_plan(app)
        return app

    def decision(self, app, kind):
        elements = app.button if kind == "apply" else app.selectbox
        matches = [item for item in elements if (item.key or "").startswith(f"fulfillment_decision_{kind}_")]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def has_decision(self, app):
        return any((item.key or "").startswith("fulfillment_decision_apply_") for item in app.button)

    def capture_downloads(self, app):
        captured = {}

        def capture(label, data, **kwargs):
            captured[kwargs.get("key")] = {
                "label": label, "data": data() if callable(data) else data,
                "file_name": kwargs.get("file_name"),
            }
            return False

        with patch("frontend.pages.data_fulfillment.st.download_button", side_effect=capture):
            app.run()
        self.assert_clean(app)
        return captured

    def csv_rows(self, data):
        return list(csv.DictReader(StringIO(data.decode("utf-8-sig"))))

    def test_guided_status_decision_requires_explicit_choice_then_a_new_run(self):
        app = self.run_a(self.app())
        committed = copy.deepcopy(app.session_state["fulfillment"]["committed"])
        self.load(app, "month_c_unseen_status.csv")
        self.run_plan(app)
        before = app.session_state["fulfillment"]
        revision = before["draft_plan"]["revision"]
        digest = plan_digest(before["draft_plan"])
        source_id = before["source"]["source_id"]
        self.assertEqual(self.decision(app, "status").value, "backordered")
        self.assertNotIn(self.decision(app, "target").value,
                         ("fulfilled", "pending", "cancelled"))
        self.assertTrue(self.decision(app, "apply").disabled)
        self.assertEqual(before["committed"], committed)
        self.decision(app, "target").set_value("pending").run()
        self.assertFalse(self.decision(app, "apply").disabled)
        self.decision(app, "apply").click().run()
        self.assert_clean(app)
        changed = app.session_state["fulfillment"]
        self.assertEqual(changed["draft_plan"]["revision"], revision + 1)
        self.assertNotEqual(plan_digest(changed["draft_plan"]), digest)
        self.assertEqual(changed["draft_plan"]["steps"][1]["status_mapping"]["backordered"], "pending")
        self.assertEqual(changed["source"]["source_id"], source_id)
        self.assertIsNone(changed["last_attempt"])
        self.assertEqual(changed["committed"], committed)
        self.assertFalse(current_result(changed))
        self.assertFalse(self.has_decision(app))
        self.run_plan(app)
        result = app.session_state["fulfillment"]["committed"]["attempt"]
        self.assertEqual(result["report"]["accepted_rows"], 1)
        self.assertEqual(result["accepted"][0]["values"][5], "pending")
        self.assertTrue(current_result(app.session_state["fulfillment"]))

    def test_replacing_source_removes_unapplied_decision_and_preserves_plan(self):
        app = self.run_a(self.app())
        self.load(app, "month_c_unseen_status.csv")
        self.run_plan(app)
        digest = plan_digest(app.session_state["fulfillment"]["draft_plan"])
        a_run = app.session_state["fulfillment"]["committed"]["attempt"]["run_id"]
        self.decision(app, "target").set_value("cancelled").run()
        self.load(app, "month_b.csv")
        workspace = app.session_state["fulfillment"]
        self.assertIsNone(workspace["last_attempt"])
        self.assertEqual(plan_digest(workspace["draft_plan"]), digest)
        self.assertEqual(workspace["committed"]["attempt"]["run_id"], a_run)
        self.assertFalse(self.has_decision(app))
        self.run_plan(app)
        self.assertEqual(app.session_state["fulfillment"]["committed"]["attempt"]["report"],
                         EXPECTED["completed_runs"]["month_b.csv"]["report"])

    def test_undo_guided_mapping_restores_source_plan_attempt_and_prior_result(self):
        app = self.run_a(self.app())
        self.load(app, "month_c_unseen_status.csv")
        self.run_plan(app)
        before = copy.deepcopy(app.session_state["fulfillment"])
        self.decision(app, "target").set_value("pending").run()
        self.decision(app, "apply").click().run()
        self.assertIsNone(app.session_state["fulfillment"]["last_attempt"])
        self.assertEqual(app.session_state["fulfillment"]["draft_plan"]["revision"],
                         before["draft_plan"]["revision"] + 1)
        app.button(key="fulfillment_undo").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"], before)
        self.assertEqual(self.decision(app, "status").value, "backordered")
        self.assertTrue(self.decision(app, "apply").disabled)
        self.assertFalse(current_result(app.session_state["fulfillment"]))

    def test_each_unknown_status_requires_its_own_explicit_decision(self):
        app = self.app()
        self.load_bytes(app, b"order_id,line_id,order_date,sku,quantity,fulfillment_status\n"
                        b"000601,01,2026-10-01,sku-1,1,backordered\n"
                        b"000602,01,2026-10-02,sku-2,2,returned\n")
        self.run_plan(app)
        initial_revision = app.session_state["fulfillment"]["draft_plan"]["revision"]
        self.decision(app, "status").set_value("backordered").run()
        self.decision(app, "target").set_value("pending").run()
        self.decision(app, "status").set_value("returned").run()
        self.assertTrue(self.decision(app, "apply").disabled)
        self.decision(app, "status").set_value("backordered").run()
        self.decision(app, "target").set_value("pending").run()
        self.decision(app, "apply").click().run()
        self.run_plan(app)
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["last_attempt"]["status"], "decision_required")
        self.assertIsNone(workspace["committed"])
        self.assertEqual(self.decision(app, "status").value, "returned")
        self.assertTrue(self.decision(app, "apply").disabled)
        self.decision(app, "target").set_value("cancelled").run()
        self.decision(app, "apply").click().run()
        self.run_plan(app)
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["draft_plan"]["revision"], initial_revision + 2)
        self.assertEqual([row["values"][5] for row in workspace["committed"]["attempt"]["accepted"]],
                         ["pending", "cancelled"])

    def test_conflicts_are_actionable_and_downloads_remain_bound_to_prior_result(self):
        app = self.run_a(self.app())
        a_run = app.session_state["fulfillment"]["committed"]["attempt"]["run_id"]
        a_source = app.session_state["fulfillment"]["committed"]["source"]["source_id"]
        self.load(app, "month_c_conflicting_duplicates.csv")
        self.run_plan(app)
        latest = app.session_state["fulfillment"]["last_attempt"]
        self.assertEqual(latest["status"], "decision_required")
        self.assertIsNone(latest["accepted"])
        text = self.rendered_text(app)
        for detail in ("000401", "01", "quantity"):
            self.assertIn(detail, text)
        self.assertIn("conflict", text.lower())
        self.assertFalse(self.has_decision(app))
        downloads = self.capture_downloads(app)
        accepted = downloads["fulfillment_accepted_download"]
        quarantined = downloads["fulfillment_rejected_download"]
        report = json.loads(downloads["fulfillment_report_download"]["data"])
        self.assertIn(a_run[:12], accepted["file_name"])
        self.assertIn(a_run[:12], quarantined["file_name"])
        self.assertEqual(report["run_id"], a_run)
        self.assertEqual(report["source_id"], a_source)
        oracle = EXPECTED["completed_runs"]["month_a.csv"]
        self.assertEqual([list(row.values()) for row in self.csv_rows(accepted["data"])],
                         [[str(value) for value in item["values"]] for item in oracle["accepted"]])
        self.assertEqual(len(self.csv_rows(quarantined["data"])), 8)
        self.assertEqual(report["report"], oracle["report"])

    def test_quarantine_filter_limits_preview_only_and_keeps_full_downloads(self):
        app = self.run_a(self.app())
        run_id = app.session_state["fulfillment"]["committed"]["attempt"]["run_id"]
        key = f"fulfillment_rejection_reason_{run_id}"
        self.assertEqual(app.selectbox(key=key).value, "All reasons")
        app.selectbox(key=key).set_value("nonpositive_quantity").run()
        self.assert_clean(app)
        quarantine = [item.value for item in app.dataframe if "reason_codes" in item.value.columns]
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(list(quarantine[0]["source_record"]), [5, 6])
        metrics = {item.label: int(item.value.replace(",", "")) for item in app.metric}
        self.assertEqual(metrics["Quarantined lines"], 8)
        self.assertEqual(metrics["Accepted lines"], 4)
        downloads = self.capture_downloads(app)
        self.assertEqual(len(self.csv_rows(downloads["fulfillment_rejected_download"]["data"])), 8)
        self.assertEqual(json.loads(downloads["fulfillment_report_download"]["data"])["report"],
                         EXPECTED["completed_runs"]["month_a.csv"]["report"])

    def test_skipped_checkpoint_is_visible_even_when_history_remains_empty(self):
        with patch("backend.resources.MAX_HISTORY_BYTES", 1):
            app = self.app()
            self.load(app, "month_a.csv")
            self.assertEqual(app.session_state["fulfillment"]["history"], [])
            self.assertTrue(app.button(key="fulfillment_undo").disabled)
            text = self.rendered_text(app).lower()
            self.assertIn("undo", text)
            self.assertTrue("budget" in text or "checkpoint" in text)
            self.assertTrue(any("undo" in item.value.lower() for item in app.warning))
            self.run_plan(app)
            self.assertEqual(app.session_state["fulfillment"]["history"], [])
            self.assertTrue(app.button(key="fulfillment_undo").disabled)
            self.assertEqual(app.session_state["fulfillment"]["committed"]["attempt"]["report"],
                             EXPECTED["completed_runs"]["month_a.csv"]["report"])

    def test_reason_filter_preserves_all_reasons_and_counts_a_line_once(self):
        app = self.app()
        self.load(app, "month_c_invalid_rows.csv")
        self.run_plan(app)
        result = app.session_state["fulfillment"]["committed"]["attempt"]
        app.selectbox(key=f"fulfillment_rejection_reason_{result['run_id']}").set_value("nonpositive_quantity").run()
        self.assert_clean(app)
        quarantine = [item.value for item in app.dataframe if "reason_codes" in item.value.columns]
        self.assertEqual(list(quarantine[0]["source_record"]), [2])
        self.assertEqual(json.loads(quarantine[0]["reason_codes"].iloc[0]),
                         ["missing_order_id", "missing_sku", "nonpositive_quantity"])
        metrics = {item.label: int(item.value.replace(",", "")) for item in app.metric}
        self.assertEqual(metrics["Quarantined lines"], 7)
        downloads = self.capture_downloads(app)
        self.assertEqual(len(self.csv_rows(downloads["fulfillment_rejected_download"]["data"])), 7)
        report = json.loads(downloads["fulfillment_report_download"]["data"])["report"]
        self.assertEqual(report["rejection_reason_assignments"], 9)
        self.assertEqual(report, EXPECTED["completed_runs"]["month_c_invalid_rows.csv"]["report"])

    def test_failed_latest_attempt_and_previous_result_survive_page_save_reopen(self):
        app = self.run_a(self.app())
        previous = copy.deepcopy(app.session_state["fulfillment"]["committed"])
        self.load(app, "month_b.csv")
        with patch("backend.fulfillment._report", side_effect=RuntimeError("private source value")):
            self.run_plan(app)
        before = copy.deepcopy(app.session_state["fulfillment"])
        self.assertEqual(before["last_attempt"]["status"], "failed")
        self.assertEqual(before["committed"], previous)
        self.assertIsNone(before["last_attempt"]["report"])
        self.assertNotIn("private source value", self.rendered_text(app))
        self.assertIn(before["source"]["source_id"][:12], self.rendered_text(app))
        self.assertIn(previous["attempt"]["run_id"][:12], self.rendered_text(app))
        app.selectbox(key="journey_page").set_value("project").run()
        app.checkbox(key="project_include_history").uncheck().run()
        app.button(key="project_prepare").click().run()
        self.assert_clean(app)
        archive = app.session_state["prepared_project_download"]["data"]
        saved = loads_project(archive)["fulfillment"]
        self.assertEqual(saved["last_attempt"], before["last_attempt"])
        self.assertEqual(saved["committed"], previous)
        self.assertEqual(saved["history"], [])
        app.button(key="journey_clear").click().run()
        with self.uploads(project_file=BytesIO(archive)):
            app.run()
            app.button(key="project_open").click().run()
        app.selectbox(key="journey_page").set_value("fulfillment").run()
        self.assert_clean(app)
        reopened = app.session_state["fulfillment"]
        self.assertEqual(reopened["last_attempt"], before["last_attempt"])
        self.assertEqual(reopened["committed"], previous)
        self.assertFalse(current_result(reopened))
        self.assertTrue(any("failed" in item.value.lower() for item in app.error))

    def test_first_failed_run_does_not_claim_a_previous_result_exists(self):
        app = self.app()
        self.load(app, "month_a.csv")
        with patch("backend.fulfillment._report", side_effect=RuntimeError("private source value")):
            self.run_plan(app)
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["last_attempt"]["status"], "failed")
        self.assertIsNone(workspace["committed"])
        self.assertEqual(len(app.metric), 0)
        text = self.rendered_text(app)
        self.assertIn("No successful result", text)
        self.assertIn("reconcile", text)
        self.assertIn("RuntimeError", text)
        self.assertNotIn("private source value", text)
        self.assertNotIn("previous successful result is retained", text)
        self.assertNotIn("last successful result is retained", text)
        downloads = self.capture_downloads(app)
        self.assertNotIn("fulfillment_accepted_download", downloads)
        self.assertNotIn("fulfillment_rejected_download", downloads)
        self.assertNotIn("fulfillment_report_download", downloads)
        self.run_plan(app)
        self.assertEqual(app.session_state["fulfillment"]["committed"]["attempt"]["report"],
                         EXPECTED["completed_runs"]["month_a.csv"]["report"])

    def test_dataset_and_fulfillment_undo_controls_restore_their_own_workspace(self):
        app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from backend.state import StateManager
from app import home_page

StateManager.init_session()
if not st.session_state.get("journey_initialized"):
    StateManager.load_dataset(pd.DataFrame({"value": [1, 1, 2]}), {})
    StateManager.apply_operation("Remove duplicates", st.session_state.df.drop_duplicates(),
        operation={"kind": "duplicates", "parameters": {"subset": ["value"]}})
    st.session_state.journey_initialized = True
home_page()
''', default_timeout=30).run(timeout=30)
        app.sidebar.selectbox[0].set_value("Order Fulfillment").run()
        self.assert_clean(app)
        self.assertTrue(any(item.value == "Dataset history" for item in app.sidebar.subheader))
        dataset_undo = [item for item in app.sidebar.button if item.label == "↩ Undo dataset change"]
        self.assertEqual(len(dataset_undo), 1)
        self.run_a(app)
        successful_a = copy.deepcopy(app.session_state["fulfillment"]["committed"])
        generic_frame = app.session_state["df"].copy(deep=True)
        generic_history_count = len(app.session_state["history"])
        self.load(app, "month_b.csv")
        self.assertFalse(current_result(app.session_state["fulfillment"]))
        app.button(key="fulfillment_undo").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"]["committed"], successful_a)
        self.assertTrue(current_result(app.session_state["fulfillment"]))
        assert_frame_equal(app.session_state["df"], generic_frame)
        self.assertEqual(len(app.session_state["history"]), generic_history_count)
        fulfillment_before_dataset_undo = copy.deepcopy(app.session_state["fulfillment"])
        [item for item in app.sidebar.button if item.label == "↩ Undo dataset change"][0].click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1, 1, 2])
        self.assertEqual(app.session_state["history"], [])
        self.assertEqual(app.session_state["fulfillment"], fulfillment_before_dataset_undo)


if __name__ == "__main__":
    unittest.main()
