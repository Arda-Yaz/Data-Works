import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = '''
import pandas as pd
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_eda import data_eda

StateManager.init_session()
if not st.session_state.get("test_initialized"):
    StateManager.load_dataset(__FRAME__, {})
    st.session_state.test_initialized = True
data_eda()
'''


class EDAPageTests(unittest.TestCase):
    def page(self, expression):
        app = AppTest.from_string(SCRIPT.replace("__FRAME__", expression)).run(timeout=30)
        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual([item.value for item in app.error], [])
        return app

    def test_all_missing_nullable_numeric_column_renders_without_crash(self):
        app = self.page('pd.DataFrame({"x": pd.Series([None, None], dtype="Int64")})')
        self.assertTrue(any("No finite numeric values" in item.value for item in app.info))

    def test_empty_and_single_row_frames_do_not_crash(self):
        for expression in ('pd.DataFrame()', 'pd.DataFrame({"x": pd.Series([], dtype="Int64")})', 'pd.DataFrame({"x": [7]})'):
            with self.subTest(expression=expression):
                self.page(expression)

    def test_numeric_zero_header_is_analyzed_and_grouped(self):
        app = self.page('pd.DataFrame({0: [1, 1, 2], 5: [2.0, 4.0, 9.0]})')
        self.assertEqual(app.selectbox(key="eda_uni_col").value, 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        app.session_state["eda_section"] = "Group-By"
        app.run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertTrue(any(table.value.columns.tolist() == ["0", "mean(0)"] for table in app.dataframe))

    def test_generated_aggregate_name_collision_renders_direct_summary(self):
        app = self.page('pd.DataFrame({"mean(x)": ["a", "a", "b"], "x": [1.0, 3.0, 9.0]})')
        app.session_state["eda_section"] = "Group-By"
        app.run()
        tables = [item.value for item in app.dataframe if "mean(x)" in item.value.columns]
        self.assertTrue(any(table.columns.is_unique and table.iloc[:, 1].tolist() == [2.0, 9.0] for table in tables))

    def test_count_column_and_unused_categories_render_only_observed_values(self):
        app = self.page('pd.DataFrame({"count": pd.Categorical(["a", "a"], categories=["a", "unused"])})')
        self.assertEqual(len(app.get("plotly_chart")), 1)
        charts = "\n".join(str(item.proto) for item in app.get("plotly_chart"))
        self.assertNotIn("unused", charts)
        app.session_state["eda_section"] = "Group-By"
        app.run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertNotIn("unused", "\n".join(str(item.proto) for item in app.get("plotly_chart")))


if __name__ == "__main__":
    unittest.main()
