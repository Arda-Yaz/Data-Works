import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from backend.experiments import project_experiments


SCRIPT = '''
import numpy as np
import pandas as pd
import streamlit as st
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_ml import data_ml

StateManager.init_session()
if not st.session_state.get("initialized"):
    frame = pd.DataFrame({"target": np.arange(40, dtype=float) * 0.3 + 0.1,
                          "x": np.arange(40, dtype=float), "other": np.arange(40, dtype=float) ** 2})
    StateManager.load_dataset(frame, infer_column_meta(frame))
    st.session_state.initialized = True
if st.button("Change data", key="test_change"):
    frame = st.session_state.df.copy()
    frame.loc[0, "x"] = 500.0
    StateManager.apply_operation("Change data", frame)
    st.rerun()
if st.button("Undo data", key="test_undo"):
    StateManager.undo()
    st.rerun()
if st.button("Replace data", key="test_replace"):
    StateManager.load_dataset(pd.DataFrame({"target": [0.1,0.2,0.3,0.4], "x": [1.0,2.0,3.0,4.0]}), {})
    st.rerun()
page = st.radio("Test page", ["ML", "Other"], key="test_page")
if page == "ML":
    data_ml()
'''


class ExperimentPageTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(SCRIPT).run(timeout=30)
        self.assert_clean(app)
        return app

    def assert_clean(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def train(self, app):
        app.multiselect(key="ml_models").set_value(["Linear Regression"])
        app.button(key="ml_train").click().run(timeout=30)
        self.assert_clean(app)

    def tables(self, app):
        return [item.value for item in app.dataframe if item.value.index.name == "Model"]

    def test_results_survive_widget_changes_and_navigation_without_retraining(self):
        app = self.app()
        self.train(app)
        identifier = app.session_state["experiments"][0]["id"]
        with patch("frontend.pages.data_ml.train_model", side_effect=AssertionError("Unexpected retraining")):
            app.slider(key="ml_split").set_value(0.3).run()
            self.assert_clean(app)
            self.assertTrue(self.tables(app))
            app.radio(key="test_page").set_value("Other").run()
            app.radio(key="test_page").set_value("ML").run()
            self.assert_clean(app)
            self.assertTrue(self.tables(app))
        self.assertEqual(app.session_state["experiments"][0]["id"], identifier)
        self.assertEqual(app.session_state["experiments"][0]["config"]["test_size"], 0.2)

    def test_failure_of_new_run_keeps_previous_completed_report_visible(self):
        app = self.app()
        self.train(app)
        identifier = app.session_state["experiments"][0]["id"]
        with patch("frontend.pages.data_ml.auto_preprocess", side_effect=ValueError("Cannot split this selection")):
            app.button(key="ml_train").click().run(timeout=30)
        self.assert_clean(app)
        self.assertTrue(any("Cannot split" in item.value for item in app.error))
        self.assertTrue(self.tables(app))
        self.assertEqual([record["id"] for record in app.session_state["experiments"]], [identifier])

    def test_data_edits_preserve_historical_run_and_undo_restores_current_association(self):
        app = self.app()
        self.train(app)
        version = app.session_state["experiments"][0]["data_version"]
        app.button(key="test_change").click().run()
        self.assert_clean(app)
        self.assertTrue(self.tables(app))
        self.assertNotEqual(app.session_state["data_version"], version)
        self.assertTrue(any("earlier dataset state" in item.value for item in app.warning))
        app.button(key="test_undo").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["data_version"], version)
        self.assertFalse(any("earlier dataset state" in item.value for item in app.warning))

    def test_reopened_report_renders_without_unpickling_or_live_models(self):
        app = self.app()
        self.train(app)
        app.session_state["experiments"] = project_experiments(app.session_state["experiments"])
        app.run()
        self.assert_clean(app)
        self.assertTrue(self.tables(app))
        self.assertTrue(any("Retrain" in item.value for item in app.info))
        self.assertTrue(app.get("plotly_chart"))

    def test_dataset_replacement_and_experiment_delete_remove_reports(self):
        app = self.app()
        self.train(app)
        app.button(key="ml_delete_experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["experiments"], [])
        self.assertFalse(self.tables(app))
        self.train(app)
        app.button(key="test_replace").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["experiments"], [])
        self.assertFalse(self.tables(app))

    def test_identical_model_plots_do_not_create_duplicate_streamlit_ids(self):
        script = SCRIPT.replace(
            '"target": np.arange(40, dtype=float) * 0.3 + 0.1',
            '"target": ["a", "b"] * 20',
        ).replace('"x": np.arange(40, dtype=float)', '"x": [0.0, 1.0] * 20')
        app = AppTest.from_string(script).run(timeout=30)
        self.assert_clean(app)
        app.multiselect(key="ml_models").set_value(["Logistic Regression", "Random Forest"])
        app.button(key="ml_train").click().run(timeout=30)
        self.assert_clean(app)
        self.assertEqual(len(app.session_state["experiments"][0]["results"]), 2)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 4)


if __name__ == "__main__":
    unittest.main()
