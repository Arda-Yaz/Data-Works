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
        app = AppTest.from_string(SCRIPT, default_timeout=30).run(timeout=30)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def test_prepared_project_includes_edits_and_is_invalidated_by_name_change(self):
        app = self.app()
        self.assertTrue(any("Current session only" in item.value for item in app.caption))
        app.button(key="test_operation").click().run()
        app.button(key="project_prepare").click().run()
        self.assert_clean(app)
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        self.assertTrue(any("cannot confirm" in item.value for item in app.caption))
        saved = loads_project(app.session_state["prepared_project_download"]["data"])
        self.assertEqual(saved["df"]["value"].tolist(), [1, 2])
        self.assertEqual(saved["original_df"]["value"].tolist(), [1, 1, 2])
        self.assertEqual(len(saved["history"]), 1)
        self.assertEqual(saved["operation_log"][0]["kind"], "duplicates")
        app.text_input(key="project_name_input").set_value("Renamed project").run()
        self.assert_clean(app)
        self.assertNotIn("prepared_project_download", app.session_state)
        self.assertTrue(any("Current session only" in item.value for item in app.caption))

    def test_optional_history_and_verification(self):
        app = self.app()
        app.button(key="test_operation").click().run()
        app.button(key="project_prepare").click().run()
        app.checkbox(key="project_include_history").uncheck().run()
        self.assertNotIn("prepared_project_download", app.session_state)
        self.assertTrue(any("Undo snapshots are omitted" in item.value for item in app.warning))
        self.assertTrue(any("Current session only" in item.value for item in app.caption))
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

    def test_same_page_open_restores_project_name_and_resets_prepared_download(self):
        app = self.app()
        app.text_input(key="project_name_input").set_value("Saved October report").run()
        app.button(key="project_prepare").click().run()
        archive = app.session_state["prepared_project_download"]["data"]
        app.text_input(key="project_name_input").set_value("Different active project").run()
        app.button(key="project_prepare").click().run()
        with patch("frontend.pages.data_project.st.file_uploader", return_value=BytesIO(archive)):
            app.run()
            app.button(key="project_open").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["project_name"], "Saved October report")
        self.assertEqual(app.text_input(key="project_name_input").value, "Saved October report")
        self.assertNotIn("prepared_project_download", app.session_state)
        self.assertTrue(any("Current session only" in item.value for item in app.caption))
        app.button(key="project_prepare").click().run()
        restored = loads_project(app.session_state["prepared_project_download"]["data"])
        self.assertEqual(restored["project_name"], "Saved October report")

    def test_fulfillment_only_project_explains_retained_and_omitted_history(self):
        app = AppTest.from_string('''
import streamlit as st
from backend.fulfillment import ingest_csv
from backend.state import StateManager
from frontend.pages.data_project import data_project
StateManager.init_session()
if not st.session_state.get("initialized"):
    StateManager.set_fulfillment_source(ingest_csv(
        b"order_id,line_id,order_date,sku,quantity,fulfillment_status\\n0001,01,2026-10-01,sku-a,2,pending\\n"))
    StateManager.run_fulfillment()
    st.session_state.initialized = True
data_project()
''').run(timeout=30)
        self.assert_clean(app)
        self.assertIsNone(app.session_state["df"])
        self.assertTrue(any("dataset 0; fulfillment 2" in item.value for item in app.caption))
        app.checkbox(key="project_include_history").uncheck().run()
        self.assertTrue(any("starts with no undo history" in item.value for item in app.warning))
        app.button(key="project_prepare").click().run()
        self.assert_clean(app)
        payload = loads_project(app.session_state["prepared_project_download"]["data"])
        self.assertEqual(payload["fulfillment"]["history"], [])
        self.assertEqual(payload["fulfillment"]["committed"]["attempt"]["report"]["accepted_rows"], 1)
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))

    def test_shared_download_status_rejects_stale_revision_or_history_option(self):
        app = AppTest.from_string('''
import streamlit as st
from frontend.project_status import show_project_download_status
if "project_revision" not in st.session_state:
    st.session_state.project_revision = 7
    st.session_state.prepared_project_download = {"token": (7, True), "data": b"prepared"}
st.checkbox("Include history", value=True, key="project_include_history")
if st.button("Edit current project", key="edit"):
    st.session_state.project_revision += 1
show_project_download_status()
''').run(timeout=30)
        self.assert_clean(app)
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        app.checkbox(key="project_include_history").uncheck().run()
        self.assertTrue(any("Current session only" in item.value for item in app.caption))
        app.checkbox(key="project_include_history").check().run()
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        app.button(key="edit").click().run()
        self.assert_clean(app)
        self.assertTrue(any("Current session only" in item.value for item in app.caption))

    def test_history_preference_and_prepared_archive_survive_page_navigation(self):
        app = AppTest.from_string('''
import streamlit as st
from backend.fulfillment import ingest_csv
from backend.state import StateManager
from frontend.pages.data_fulfillment import data_fulfillment
from frontend.pages.data_project import data_project
StateManager.init_session()
if not st.session_state.get("initialized"):
    StateManager.set_fulfillment_source(ingest_csv(
        b"order_id,line_id,order_date,sku,quantity,fulfillment_status\\n0001,01,2026-10-01,sku-a,2,pending\\n"))
    StateManager.run_fulfillment()
    st.session_state.initialized = True
page = st.selectbox("Page", ["project", "fulfillment"], key="test_page")
if page == "project":
    data_project()
else:
    data_fulfillment()
''', default_timeout=30).run(timeout=30)
        self.assert_clean(app)
        app.checkbox(key="project_include_history").uncheck().run()
        app.button(key="project_prepare").click().run()
        archive = app.session_state["prepared_project_download"]["data"]
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        app.selectbox(key="test_page").set_value("fulfillment").run()
        app.run()
        self.assert_clean(app)
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        self.assertTrue(any("cannot confirm" in item.value for item in app.caption))
        app.selectbox(key="test_page").set_value("project").run()
        self.assert_clean(app)
        self.assertFalse(app.checkbox(key="project_include_history").value)
        self.assertTrue(any("Current archive prepared" in item.value for item in app.caption))
        self.assertEqual(app.session_state["prepared_project_download"]["data"], archive)
        self.assertEqual(loads_project(archive)["fulfillment"]["history"], [])

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
