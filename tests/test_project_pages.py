from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from backend.project_io import loads_project


SCRIPT = '''
import pandas as pd
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_project import data_project

StateManager.init_session()
if not st.session_state.get("initialized"):
    StateManager.load_dataset(pd.DataFrame({"value": [1, 1, 2], "text": ["=SUM(A1)", "a", "b"]}), {})
    st.session_state.initialized = True
if st.button("Remove duplicates", key="test_operation"):
    StateManager.apply_operation("Remove duplicates", st.session_state.df.drop_duplicates(subset=["value"]),
        operation={"kind": "duplicates", "parameters": {"subset": ["value"]}})
    st.rerun()
if st.button("Empty session", key="test_clear"):
    StateManager.clear()
    st.rerun()
data_project()
'''


class ProjectPageTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(SCRIPT).run(timeout=30)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def test_prepared_project_includes_edits_and_is_invalidated_by_name_change(self):
        app = self.app()
        app.button(key="test_operation").click().run()
        app.button(key="project_prepare").click().run()
        self.assert_clean(app)
        saved = loads_project(app.session_state["prepared_project_download"]["data"])
        self.assertEqual(saved["df"]["value"].tolist(), [1, 2])
        self.assertEqual(saved["original_df"]["value"].tolist(), [1, 1, 2])
        self.assertEqual(len(saved["history"]), 1)
        self.assertEqual(saved["operation_log"][0]["kind"], "duplicates")
        app.text_input(key="project_name_input").set_value("Renamed project").run()
        self.assert_clean(app)
        self.assertNotIn("prepared_project_download", app.session_state)

    def test_optional_history_and_verification(self):
        app = self.app()
        app.button(key="test_operation").click().run()
        app.checkbox(key="project_include_history").uncheck().run()
        app.button(key="project_prepare").click().run()
        saved = loads_project(app.session_state["prepared_project_download"]["data"])
        self.assertEqual(saved["history"], [])
        app.button(key="project_verify").click().run()
        self.assert_clean(app)
        self.assertTrue(any("reproduce" in item.value for item in app.success))

    def test_invalid_archive_leaves_data_and_history_unchanged(self):
        app = self.app()
        app.button(key="test_operation").click().run()
        before = app.session_state["df"].copy(deep=True)
        version = app.session_state["data_version"]
        with patch("frontend.pages.data_project.st.file_uploader", return_value=BytesIO(b"invalid zip")):
            app.run()
            app.button(key="project_open").click().run()
        self.assert_clean(app)
        self.assertTrue(any("unchanged" in item.value for item in app.error))
        assert_frame_equal(app.session_state["df"], before)
        self.assertEqual(app.session_state["data_version"], version)
        self.assertEqual(len(app.session_state["history"]), 1)

    def test_project_can_open_into_empty_session(self):
        app = self.app()
        app.button(key="project_prepare").click().run()
        archive = app.session_state["prepared_project_download"]["data"]
        app.button(key="test_clear").click().run()
        self.assert_clean(app)
        self.assertIsNone(app.session_state["df"])
        with patch("frontend.pages.data_project.st.file_uploader", return_value=BytesIO(archive)):
            app.run()
            app.button(key="project_open").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1, 1, 2])
        self.assertEqual(app.radio(key="export_csv_mode").value, "spreadsheet_safe")

    def test_project_route_is_accessible_without_an_upload(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        app.sidebar.selectbox[0].set_value("Project & Exports").run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Project & Exports")
        self.assertTrue(app.button(key="project_open").disabled)

    def test_mismatch_recipe_records_cleared_rules(self):
        app = AppTest.from_string('''
import pandas as pd
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_cleaning import data_cleaning
StateManager.init_session()
if not st.session_state.get("initialized"):
    StateManager.load_dataset(pd.DataFrame({"a": ["x", "y"], "b": ["keep", "bad"]}), {})
    StateManager.update_column_meta("a", allowed_values=["x", "y"])
    st.session_state.initialized = True
data_cleaning()
''').run(timeout=30)
        app.text_input(key="av_a").set_value("")
        app.text_input(key="av_b").set_value("keep")
        app.button(key="find_mismatch").click().run()
        app.button(key="apply_mismatch").click().run()
        self.assert_clean(app)
        from backend.operations import replay_operations
        result = replay_operations(app.session_state["original_df"], app.session_state["original_column_meta"], app.session_state["operation_log"])
        assert_frame_equal(result["df"], app.session_state["df"])
        self.assertEqual(result["column_meta"], app.session_state["column_meta"])
        self.assertIsNone(result["column_meta"]["a"]["allowed_values"])


if __name__ == "__main__":
    unittest.main()
