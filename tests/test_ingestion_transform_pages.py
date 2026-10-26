from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from backend.operations import replay_operations
from backend.project_io import dumps_project, loads_project


UPLOAD = '''
from backend.state import StateManager
from frontend.pages.data_upload import data_upload
StateManager.init_session()
data_upload()
'''

CLEANING = '''
import pandas as pd
import streamlit as st
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_cleaning import data_cleaning
StateManager.init_session()
if not st.session_state.get("initialized"):
    frame = pd.DataFrame({"label": [" A ", "B", None], "id": ["001", "002", "003"]}, index=[7,7,9])
    meta = infer_column_meta(frame)
    meta["label"]["allowed_values"] = [" A ", "B"]
    StateManager.load_dataset(frame, meta)
    st.session_state.initialized = True
if st.button("Test undo", key="test_undo"):
    StateManager.undo()
    st.rerun()
if st.button("Save test project", key="test_project"):
    st.session_state.saved_test_project = StateManager.export_project()
data_cleaning()
'''


def upload(data, name):
    stream = BytesIO(data if isinstance(data, bytes) else data.encode())
    stream.name = name
    return stream


class IngestionAndTransformPageTests(unittest.TestCase):
    def clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def test_upload_waits_for_options_then_converts_localized_numbers_only_on_confirmation(self):
        source = upload("customer_id;city;amount\n0001;İzmir;1.234,50\n0002;Ankara;2.345,75\n".encode("cp1254"), "data.csv")
        with patch("frontend.pages.data_upload.st.file_uploader", return_value=source):
            app = AppTest.from_string(UPLOAD).run()
            self.assertIsNone(app.session_state["df"])
            app.selectbox(key="ingest_delimiter").set_value("Semicolon")
            app.text_input(key="ingest_encoding").set_value("cp1254")
            app.selectbox(key="ingest_decimal").set_value(",")
            app.selectbox(key="ingest_thousands").set_value(".")
            app.button(key="ingest_load").click().run()
            self.clean(app)
            self.assertEqual(app.session_state["df"]["amount"].tolist(), ["1.234,50", "2.345,75"])
            self.assertEqual(app.session_state["df"]["customer_id"].tolist(), ["0001", "0002"])
            next(button for button in app.button if button.label == "Confirm Schema").click().run()
        self.clean(app)
        self.assertEqual(app.session_state["df"]["amount"].tolist(), [1234.5, 2345.75])
        self.assertEqual(app.session_state["original_df"]["amount"].tolist(), ["1.234,50", "2.345,75"])
        replayed = replay_operations(app.session_state["original_df"], app.session_state["original_column_meta"], app.session_state["operation_log"])
        assert_frame_equal(replayed["df"], app.session_state["df"])

    def test_excel_sheet_control_reads_the_chosen_sheet(self):
        source = BytesIO()
        with pd.ExcelWriter(source, engine="openpyxl") as writer:
            pd.DataFrame({"code": ["001"]}).to_excel(writer, index=False, sheet_name="First")
            pd.DataFrame({"code": ["002"]}).to_excel(writer, index=False, sheet_name="Second")
        source.name = "book.xlsx"
        with patch("frontend.pages.data_upload.st.file_uploader", return_value=source):
            app = AppTest.from_string(UPLOAD).run()
            app.selectbox(key="ingest_sheet").set_value("Second")
            app.button(key="ingest_load").click().run()
        self.clean(app)
        self.assertEqual(app.session_state["df"]["code"].tolist(), ["002"])

    def test_json_object_path_and_lines_controls(self):
        examples = [
            (upload('{"data":{"rows":[{"code":"001"}]}}', "data.json"), '["data", "rows"]'),
            (upload('{"code":"002"}\n{"code":"003"}\n', "data.jsonl"), "[]"),
        ]
        for source, path in examples:
            with self.subTest(name=source.name), patch("frontend.pages.data_upload.st.file_uploader", return_value=source):
                app = AppTest.from_string(UPLOAD).run()
                app.text_input(key="ingest_json_path").set_value(path)
                app.button(key="ingest_load").click().run()
                self.clean(app)
                expected = ["001"] if source.name.endswith(".json") else ["002", "003"]
                self.assertEqual(app.session_state["df"]["code"].tolist(), expected)

    def test_invalid_encoding_and_json_path_leave_the_session_empty(self):
        source = upload(b"name\n\xff\n", "bad.csv")
        with patch("frontend.pages.data_upload.st.file_uploader", return_value=source):
            app = AppTest.from_string(UPLOAD).run()
            app.button(key="ingest_load").click().run()
        self.clean(app)
        self.assertTrue(app.error)
        self.assertIsNone(app.session_state["df"])
        self.assertEqual(app.session_state["history"], [])
        with patch("frontend.pages.data_upload.st.file_uploader", return_value=upload("{}", "bad.json")):
            app = AppTest.from_string(UPLOAD).run()
            app.text_input(key="ingest_json_path").set_value("data.rows").run()
        self.clean(app)
        self.assertTrue(app.error)
        self.assertIsNone(app.session_state["df"])

    def test_text_preview_apply_and_project_recipe_agree_and_are_undoable(self):
        app = AppTest.from_string(CLEANING).run()
        before = app.session_state["df"].copy(deep=True)
        app.multiselect(key="text_columns").set_value(["label"])
        app.checkbox(key="text_strip").check()
        app.selectbox(key="text_case").set_value("lower").run()
        app.button(key="text_preview").click().run()
        self.clean(app)
        assert_frame_equal(app.session_state["df"], before)
        self.assertEqual(app.session_state["history"], [])
        app.button(key="text_apply").click().run()
        self.clean(app)
        self.assertEqual(app.session_state["df"]["label"].iloc[:2].tolist(), ["a", "b"])
        self.assertTrue(pd.isna(app.session_state["df"]["label"].iloc[2]))
        self.assertEqual(app.session_state["column_meta"]["label"]["allowed_values"], ["a", "b"])
        self.assertEqual(app.session_state["df"].index.tolist(), [7, 7, 9])
        self.assertEqual(app.session_state["df"]["id"].tolist(), ["001", "002", "003"])
        app.button(key="test_project").click().run()
        saved = loads_project(dumps_project(app.session_state["saved_test_project"]))
        replayed = replay_operations(saved["original_df"], saved["original_column_meta"], saved["operation_log"])
        assert_frame_equal(replayed["df"], saved["df"])
        self.assertEqual(replayed["column_meta"], saved["column_meta"])
        app.button(key="test_undo").click().run()
        self.clean(app)
        assert_frame_equal(app.session_state["df"], before)
        self.assertEqual(app.session_state["operation_log"], [])

    def test_changed_text_controls_require_a_new_preview(self):
        app = AppTest.from_string(CLEANING).run()
        app.multiselect(key="text_columns").set_value(["label"])
        app.checkbox(key="text_strip").check().run()
        app.button(key="text_preview").click().run()
        self.assertFalse(app.button(key="text_apply").disabled)
        app.selectbox(key="text_case").set_value("upper").run()
        self.clean(app)
        self.assertTrue(app.button(key="text_apply").disabled)
        self.assertEqual(app.session_state["history"], [])

    def test_category_mapping_uses_the_same_preview_and_replay_boundary(self):
        app = AppTest.from_string(CLEANING).run()
        app.radio(key="text_mode").set_value("Map categories").run()
        mapping = pd.DataFrame({"Original": [" A ", "B"], "Replacement": ["B", " A "]})
        with patch("frontend.pages.data_cleaning.st.data_editor", return_value=mapping):
            app.run()
            app.button(key="text_preview").click().run()
            app.button(key="text_apply").click().run()
        self.clean(app)
        self.assertEqual(app.session_state["df"]["label"].iloc[:2].tolist(), ["B", " A "])
        self.assertEqual(app.session_state["operation_log"][-1]["kind"], "category_map")
        replayed = replay_operations(app.session_state["original_df"], app.session_state["original_column_meta"], app.session_state["operation_log"])
        assert_frame_equal(replayed["df"], app.session_state["df"])


if __name__ == "__main__":
    unittest.main()
