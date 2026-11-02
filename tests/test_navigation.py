"""Navigation journeys preserve data while separating the two workspaces."""

from pathlib import Path
import unittest

from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
POPULATED = '''
import pandas as pd
import streamlit as st
from app import home_page
from backend.state import StateManager

StateManager.init_session()
if not st.session_state.get("navigation_fixture"):
    StateManager.load_dataset(pd.DataFrame({"quantity": [1, 2, 3], "sku": ["A", "B", "C"]}), {})
    st.session_state.navigation_fixture = True
home_page()
'''


class NavigationTests(unittest.TestCase):
    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def test_empty_workspace_guides_upload_and_allows_project_opening(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Load & review")
        self.assertEqual(app.sidebar.selectbox(key="workspace").value, "Data preparation")
        self.assertTrue(app.sidebar.button(key="nav_clean").disabled)
        self.assertTrue(app.sidebar.button(key="nav_explore").disabled)
        self.assertTrue(app.sidebar.button(key="nav_models").disabled)
        app.sidebar.button(key="nav_project").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Project & Exports")
        self.assertTrue(app.button(key="project_open").disabled)
        app.sidebar.button(key="nav_back").click().run()
        self.assertEqual(app.title[0].value, "Load & review")

    def test_workspace_and_project_navigation_preserve_data_and_last_step(self):
        app = AppTest.from_string(POPULATED, default_timeout=30).run()
        self.assert_clean(app)
        before = app.session_state["df"].copy(deep=True)
        dataset_id = app.session_state["dataset_id"]
        app.sidebar.button(key="nav_clean").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Data Cleaning")
        app.sidebar.selectbox(key="workspace").set_value("Order Fulfillment").run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Order Fulfillment")
        self.assertTrue(app.button(key="fulfillment_run").disabled)
        app.sidebar.button(key="nav_project").click().run()
        app.sidebar.button(key="nav_back").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Order Fulfillment")
        app.sidebar.selectbox(key="workspace").set_value("Data preparation").run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Data Cleaning")
        assert_frame_equal(app.session_state["df"], before)
        self.assertEqual(app.session_state["dataset_id"], dataset_id)

    def test_charts_and_models_remain_accessible_without_replacing_data(self):
        app = AppTest.from_string(POPULATED, default_timeout=30).run()
        before = app.session_state["df"].copy(deep=True)
        app.sidebar.button(key="nav_charts").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Visualization")
        app.sidebar.button(key="nav_models").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Machine Learning")
        assert_frame_equal(app.session_state["df"], before)

    def test_returning_after_dataset_is_cleared_resets_the_remembered_step(self):
        script = POPULATED.replace("home_page()", '''
if st.button("Clear dataset fixture", key="navigation_clear"):
    StateManager.clear_dataset()
home_page()
''')
        app = AppTest.from_string(script, default_timeout=30).run()
        app.sidebar.button(key="nav_clean").click().run()
        app.sidebar.selectbox(key="workspace").set_value("Order Fulfillment").run()
        app.button(key="navigation_clear").click().run()
        self.assert_clean(app)
        self.assertIsNone(app.session_state["df"])
        app.sidebar.selectbox(key="workspace").set_value("Data preparation").run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Load & review")
        self.assertEqual(app.session_state["_navigation_page"], "Upload")
        self.assertEqual(app.session_state["_last_data_page"], "Upload")
        self.assertTrue(app.sidebar.button(key="nav_clean").disabled)

    def test_project_back_route_is_normalized_after_dataset_replacement(self):
        script = POPULATED.replace("home_page()", '''
if st.button("Clear dataset fixture", key="navigation_clear"):
    StateManager.clear_dataset()
home_page()
''')
        app = AppTest.from_string(script, default_timeout=30).run()
        app.sidebar.button(key="nav_clean").click().run()
        app.sidebar.button(key="nav_project").click().run()
        app.button(key="navigation_clear").click().run()
        self.assert_clean(app)
        app.sidebar.button(key="nav_back").click().run()
        self.assert_clean(app)
        self.assertEqual(app.title[0].value, "Load & review")
        self.assertEqual(app.session_state["_navigation_page"], "Upload")


if __name__ == "__main__":
    unittest.main()
