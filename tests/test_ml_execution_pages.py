import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from backend.preprocessing import auto_preprocess


SCRIPT = '''
import numpy as np
import pandas as pd
import streamlit as st
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_ml import data_ml

StateManager.init_session()
if not st.session_state.get("test_initialized"):
    df = __FRAME__
    StateManager.load_dataset(df, infer_column_meta(df))
    __SETUP__
    st.session_state.test_initialized = True
data_ml()
'''


class MLExecutionPageTests(unittest.TestCase):
    def make_app(self, frame, setup="pass"):
        source = SCRIPT.replace("__FRAME__", frame).replace("__SETUP__", setup)
        app = AppTest.from_string(source).run(timeout=30)
        self.assert_no_exceptions(app)
        return app

    def assert_no_exceptions(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def train(self, app, models):
        app.multiselect(key="ml_models").set_value(models)
        app.button(key="ml_train").click().run(timeout=30)
        self.assert_no_exceptions(app)
        return app

    @staticmethod
    def result_tables(app):
        return [table.value for table in app.dataframe if "Model" == table.value.index.name]

    def test_multiclass_logistic_training_renders_actual_labels(self):
        app = self.make_app(
            'pd.DataFrame({"target": ["class_red", "class_green", "class_blue"] * 20, '
            '"value": np.arange(60, dtype=float), "signal": [0.0, 1.0, 2.0] * 20})'
        )
        self.assertEqual(app.selectbox(key="ml_split_strategy").value, "stratified")
        self.train(app, ["Logistic Regression"])
        self.assertEqual([item.value for item in app.error], [])
        self.assertTrue(any("Logistic Regression" in table.index for table in self.result_tables(app)))
        charts = app.get("plotly_chart")
        self.assertGreaterEqual(len(charts), 2)
        rendered = "\n".join(str(chart.proto) for chart in charts)
        for label in ("class_red", "class_green", "class_blue"):
            self.assertIn(label, rendered)

    def test_regression_uses_random_split_and_renders_results(self):
        app = self.make_app(
            'pd.DataFrame({"target": np.arange(40, dtype=float) * 0.3 + 1.1, '
            '"value": np.arange(40, dtype=float)})'
        )
        self.assertEqual(app.radio(key="ml_ptype").value, "regression")
        self.assertEqual(app.selectbox(key="ml_split_strategy").value, "random")
        self.train(app, ["Linear Regression"])
        self.assertEqual([item.value for item in app.error], [])
        self.assertTrue(any("Linear Regression" in table.index for table in self.result_tables(app)))
        self.assertGreaterEqual(len(app.get("plotly_chart")), 1)

    def test_rare_class_stratification_error_is_actionable(self):
        app = self.make_app(
            'pd.DataFrame({"target": ["common"] * 39 + ["rare"], "value": np.arange(40, dtype=float)})'
        )
        self.train(app, ["Logistic Regression"])
        errors = " ".join(item.value for item in app.error).lower()
        self.assertTrue("class" in errors or "stratif" in errors, errors)
        self.assertEqual(self.result_tables(app), [])

    def test_changing_target_resets_task_and_model_choices(self):
        app = self.make_app(
            'pd.DataFrame({"class_target": ["a", "b"] * 20, '
            '"numeric_target": np.arange(40, dtype=float) * 0.25 + 0.1, '
            '"value": np.arange(40, dtype=float)})'
        )
        app.multiselect(key="ml_models").set_value(["Logistic Regression"]).run()
        app.selectbox(key="ml_target").set_value("numeric_target").run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.radio(key="ml_ptype").value, "regression")
        self.assertNotIn("Logistic Regression", app.multiselect(key="ml_models").options)
        self.assertNotIn("numeric_target", app.multiselect(key="ml_features").value)
        self.assertEqual(app.selectbox(key="ml_split_strategy").value, "random")

    def test_grouped_classification_excludes_group_identifier(self):
        app = self.make_app(
            'pd.DataFrame({"target": ["a", "b"] * 40, "value": np.arange(80, dtype=float), '
            '"entity_id": np.repeat(np.arange(20), 4)})'
        )
        app.selectbox(key="ml_split_strategy").set_value("group").run()
        self.assert_no_exceptions(app)
        app.selectbox(key="ml_group_col").set_value("entity_id").run()
        self.assert_no_exceptions(app)
        self.assertNotIn("entity_id", app.multiselect(key="ml_features").value)
        self.train(app, ["Logistic Regression"])
        self.assertEqual([item.value for item in app.error], [])
        self.assertTrue(self.result_tables(app))

    def test_chronological_regression_excludes_time_column(self):
        app = self.make_app(
            'pd.DataFrame({"target": np.arange(60, dtype=float) * 0.5 + 0.1, '
            '"value": np.arange(60, dtype=float), '
            '"observed_at": pd.date_range("2024-01-01", periods=60, freq="D")})'
        )
        app.selectbox(key="ml_split_strategy").set_value("chronological").run()
        self.assert_no_exceptions(app)
        app.selectbox(key="ml_time_col").set_value("observed_at").run()
        self.assert_no_exceptions(app)
        self.assertNotIn("observed_at", app.multiselect(key="ml_features").value)
        self.train(app, ["Linear Regression"])
        self.assertEqual([item.value for item in app.error], [])
        self.assertTrue(self.result_tables(app))

    def test_prior_statistical_cleaning_defaults_to_original_source(self):
        app = self.make_app(
            'pd.DataFrame({"target": np.arange(40, dtype=float) * 0.3 + 0.1, '
            '"value": np.arange(40, dtype=float)})',
            setup='st.session_state.statistical_operations = ["Handle missing values"]',
        )
        self.assertEqual(app.selectbox(key="ml_data_source").value, "original")
        app.selectbox(key="ml_data_source").set_value("working").run()
        self.assert_no_exceptions(app)
        warnings = " ".join(item.value for item in app.warning).lower()
        self.assertTrue(any(word in warnings for word in ("leak", "clean", "explorat")), warnings)

    def test_small_valid_regression_still_runs_when_another_model_fails(self):
        app = self.make_app('pd.DataFrame({"target": [0.1, 0.4, 0.8, 1.5], "value": [1.0, 2.0, 3.0, 4.0]})')
        app.slider(key="ml_split").set_value(0.5)
        self.train(app, ["Linear Regression", "KNN"])
        self.assertTrue(any("Linear Regression" in table.index for table in self.result_tables(app)))
        self.assertTrue(any("KNN" in item.value for item in app.error))

    def test_changing_group_column_restores_intended_features(self):
        app = self.make_app(
            'pd.DataFrame({"target": ["a", "b"] * 40, "x": np.arange(80, dtype=float), '
            '"z": np.arange(80, dtype=float) * 2, "entity_id": np.repeat(np.arange(20), 4)})'
        )
        app.multiselect(key="ml_features").set_value(["x"]).run()
        app.selectbox(key="ml_split_strategy").set_value("group").run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.selectbox(key="ml_group_col").value, "entity_id")
        app.selectbox(key="ml_group_col").set_value("x").run()
        self.assert_no_exceptions(app)
        app.selectbox(key="ml_group_col").set_value("entity_id").run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.multiselect(key="ml_features").value, ["x"])

    def test_original_modeling_source_skips_previous_global_fill(self):
        app = self.make_app(
            'pd.DataFrame({"target": np.arange(40, dtype=float) * 0.3 + 0.1, '
            '"value": [float(i) if i % 2 == 0 else None for i in range(40)]})',
            setup='StateManager.apply_operation("Global median fill", df.fillna({"value": 999.0}), learns_from_data=True)',
        )
        self.assertEqual(app.selectbox(key="ml_data_source").value, "original")
        self.assertEqual(int(app.session_state["df"]["value"].isna().sum()), 0)
        with patch("frontend.pages.data_ml.auto_preprocess", wraps=auto_preprocess) as preprocess:
            self.train(app, ["Linear Regression"])
            self.assertEqual(int(preprocess.call_args.args[0]["value"].isna().sum()), 20)
        self.assertTrue(self.result_tables(app))


if __name__ == "__main__":
    unittest.main()
