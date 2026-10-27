"""Streamlit workflow checks over the independently specified fulfillment data."""

from contextlib import contextmanager
import copy
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from backend.exports import export_csv
from backend.fulfillment import default_plan, dumps_plan, plan_digest
from backend.project_io import loads_project
from frontend.pages.data_fulfillment import _accepted_frame, _mapping_from_json, _rejected_frame


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"
SCRIPT = '''
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_fulfillment import data_fulfillment
from frontend.pages.data_project import data_project

StateManager.init_session()
page = st.selectbox("Test page", ["fulfillment", "project"], key="test_page")
if st.button("Clear test session", key="test_clear"):
    StateManager.clear()
    st.rerun()
if page == "fulfillment":
    data_fulfillment()
else:
    data_project()
'''


class FulfillmentPageTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run(timeout=30)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    @contextmanager
    def uploads(self, **values):
        with patch("frontend.pages.data_fulfillment.st.file_uploader",
                   side_effect=lambda *args, **kwargs: values.get(kwargs.get("key"))):
            yield

    def load(self, app, name):
        contents = BytesIO((FIXTURES / name).read_bytes())
        with self.uploads(fulfillment_source_file=contents):
            app.run()
            app.button(key="fulfillment_load").click().run()
        self.assert_clean(app)
        return app

    def run_a(self, app):
        self.load(app, "month_a.csv")
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)
        return app

    def test_route_requires_no_generic_dataset_and_run_waits_for_a_source(self):
        app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
        app.sidebar.selectbox[0].set_value("Order Fulfillment").run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Order Fulfillment")
        self.assertIsNone(app.session_state["df"])
        self.assertTrue(app.button(key="fulfillment_run").disabled)

    def test_load_run_reuse_and_stale_result_are_explicit(self):
        app = self.run_a(self.app())
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["committed"]["attempt"]["report"]["accepted_rows"], 4)
        original_plan_digest = plan_digest(workspace["draft_plan"])
        a_run = workspace["committed"]["attempt"]["run_id"]
        self.load(app, "month_b.csv")
        self.assertEqual(app.session_state["fulfillment"]["committed"]["attempt"]["run_id"], a_run)
        self.assertTrue(any("earlier source or plan" in item.value for item in app.warning))
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["committed"]["attempt"]["report"]["accepted_rows"], 5)
        self.assertEqual(plan_digest(workspace["draft_plan"]), original_plan_digest)
        self.assertFalse(any("earlier source or plan" in item.value for item in app.warning))

    def test_unseen_status_preserves_previous_result_until_mapping_is_applied(self):
        app = self.run_a(self.app())
        original = copy.deepcopy(app.session_state["fulfillment"]["committed"])
        self.load(app, "month_c_unseen_status.csv")
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)
        workspace = app.session_state["fulfillment"]
        self.assertEqual(workspace["last_attempt"]["status"], "decision_required")
        self.assertEqual(workspace["committed"], original)
        self.assertTrue(any("decision is required" in item.value for item in app.warning))
        before_revision = workspace["draft_plan"]["revision"]
        mapping = copy.deepcopy(workspace["draft_plan"]["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        app.text_area[0].set_value(json.dumps(mapping))
        app.button(key="fulfillment_mapping_apply").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"]["draft_plan"]["revision"], before_revision + 1)
        self.assertEqual(app.session_state["fulfillment"]["committed"], original)
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)
        current = app.session_state["fulfillment"]["committed"]["attempt"]
        self.assertEqual(current["accepted"][0]["values"][5], "pending")
        self.assertEqual(current["report"]["accepted_rows"], 1)

    def test_invalid_source_and_plan_leave_previous_workspace_intact(self):
        app = self.run_a(self.app())
        before = copy.deepcopy(app.session_state["fulfillment"])
        with self.uploads(fulfillment_source_file=BytesIO(b'order_id,line_id\n"unterminated')):
            app.run()
            app.button(key="fulfillment_load").click().run()
        self.assert_clean(app)
        self.assertTrue(any("previous source is unchanged" in item.value for item in app.error))
        self.assertEqual(app.session_state["fulfillment"], before)
        with self.uploads(fulfillment_plan_file=BytesIO(b'{"format":"not-a-plan"}')):
            app.run()
            app.button(key="fulfillment_plan_open").click().run()
        self.assert_clean(app)
        self.assertTrue(any("previous plan is unchanged" in item.value for item in app.error))
        self.assertEqual(app.session_state["fulfillment"], before)

    def test_saved_plan_import_preserves_its_identity_and_noop_mapping_does_not_revise(self):
        app = self.run_a(self.app())
        imported_plan = default_plan()
        with self.uploads(fulfillment_plan_file=BytesIO(dumps_plan(imported_plan))):
            app.run()
            app.button(key="fulfillment_plan_open").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"]["draft_plan"], imported_plan)
        self.assertTrue(any("earlier source or plan" in item.value for item in app.warning))
        before = copy.deepcopy(app.session_state["fulfillment"])
        app.button(key="fulfillment_mapping_apply").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"], before)
        self.assertTrue(any("mapping is unchanged" in item.value for item in app.info))

    def test_mapping_editor_rejects_duplicate_keys_nonfinite_and_oversized_json(self):
        for value in ('{"pending":"pending","pending":"fulfilled"}',
                      '{"pending":NaN}', '{"pending":Infinity}', "[]", " " * (1024 * 1024 + 1)):
            with self.subTest(value=value[:80]), self.assertRaises(ValueError):
                _mapping_from_json(value)
        app = self.run_a(self.app())
        before = copy.deepcopy(app.session_state["fulfillment"])
        app.text_area[0].set_value('{"pending":"pending","pending":"fulfilled"}')
        app.button(key="fulfillment_mapping_apply").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["fulfillment"], before)
        self.assertTrue(any("Mapping was not applied" in item.value for item in app.error))

    def test_all_rejected_result_is_completed_and_has_empty_accepted_columns(self):
        app = self.app()
        data = b"order_id,line_id,order_date,sku,quantity,fulfillment_status\n000001,01,2026-10-01,sku-1,0,pending\n"
        with self.uploads(fulfillment_source_file=BytesIO(data)):
            app.run()
            app.button(key="fulfillment_load").click().run()
        app.button(key="fulfillment_run").click().run()
        self.assert_clean(app)
        attempt = app.session_state["fulfillment"]["committed"]["attempt"]
        self.assertEqual(attempt["status"], "completed")
        self.assertEqual(attempt["accepted"], [])
        self.assertEqual(attempt["report"]["rejected_rows"], 1)
        empty_csv = list(csv.reader(StringIO(export_csv(_accepted_frame([]), spreadsheet_safe=False).decode("utf-8-sig"))))
        self.assertEqual(empty_csv, [["order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status"]])

    def test_fulfillment_only_project_saves_and_reopens_without_a_generic_frame(self):
        app = self.run_a(self.app())
        before = copy.deepcopy(app.session_state["fulfillment"]["committed"])
        app.selectbox(key="test_page").set_value("project").run()
        app.button(key="project_prepare").click().run()
        self.assert_clean(app)
        archive = app.session_state["prepared_project_download"]["data"]
        payload = loads_project(archive)
        self.assertEqual(payload["mode"], "fulfillment-only")
        self.assertEqual(payload["fulfillment"]["committed"], before)
        self.assertIsNone(app.session_state["df"])
        app.button(key="test_clear").click().run()
        with self.uploads(project_file=BytesIO(archive)):
            app.run()
            app.button(key="project_open").click().run()
        self.assert_clean(app)
        self.assertIsNone(app.session_state["df"])
        self.assertEqual(app.session_state["fulfillment"]["committed"], before)
        self.assertFalse(app.button(key="project_prepare").disabled)
        app.selectbox(key="test_page").set_value("fulfillment").run()
        self.assert_clean(app)
        self.assertTrue(any("Latest attempt completed" in item.value for item in app.success))

    def test_invalid_project_preserves_fulfillment_only_committed_evidence(self):
        app = self.run_a(self.app())
        before = copy.deepcopy(app.session_state["fulfillment"])
        app.selectbox(key="test_page").set_value("project").run()
        with self.uploads(project_file=BytesIO(b"not a project")):
            app.run()
            app.button(key="project_open").click().run()
        self.assert_clean(app)
        self.assertTrue(any("current session is unchanged" in item.value for item in app.error))
        self.assertEqual(app.session_state["fulfillment"], before)

    def test_export_frames_preserve_identifier_tokens_integer_precision_and_reasons(self):
        rows = [{"values": ["000001", "01", "2026-10-01", "SKU-1", 2**63 - 1, "pending"]}]
        values = list(csv.reader(StringIO(export_csv(_accepted_frame(rows), spreadsheet_safe=False).decode("utf-8-sig"))))
        self.assertEqual(values[1], ["000001", "01", "2026-10-01", "SKU-1", "9223372036854775807", "pending"])
        raw = {"order_id": "000001", "line_id": "01", "order_date": "2026-10-01", "sku": "=private", "quantity": "0", "fulfillment_status": "pending"}
        rejection = {"source_record": 1, "row_id": "source:1", "reasons": ["nonpositive_quantity"],
                     "issues": [{"code": "nonpositive_quantity", "stage": "validate_rows", "columns": ["quantity"]}], "raw": raw}
        exported = list(csv.DictReader(StringIO(export_csv(_rejected_frame([rejection]), spreadsheet_safe=False).decode("utf-8-sig"))))[0]
        self.assertEqual({column: exported[column] for column in raw}, raw)
        self.assertEqual(json.loads(exported["reason_codes"]), ["nonpositive_quantity"])
        self.assertEqual(json.loads(exported["validation_details"]), rejection["issues"])
        duplicate = {**rejection, "source_record": 2, "row_id": "source:2", "reasons": ["exact_duplicate"],
                     "issues": [{"code": "exact_duplicate", "stage": "deduplicate", "columns": ["order_id", "line_id"]}],
                     "duplicate_of_source_record": 1, "duplicate_of_row_id": "source:1"}
        duplicate_csv = list(csv.DictReader(StringIO(export_csv(_rejected_frame([duplicate]), spreadsheet_safe=False).decode("utf-8-sig"))))[0]
        self.assertEqual(duplicate_csv["duplicate_of_source_record"], "1")
        self.assertEqual(duplicate_csv["duplicate_of_row_id"], "source:1")
        safe_csv = list(csv.DictReader(StringIO(export_csv(_rejected_frame([duplicate]), spreadsheet_safe=True).decode("utf-8-sig"))))[0]
        self.assertEqual(safe_csv["sku"], "'=private")


if __name__ == "__main__":
    unittest.main()
