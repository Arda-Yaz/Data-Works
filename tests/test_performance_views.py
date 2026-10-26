import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from backend import visualization as viz
from backend.eda import correlation_matrix, univariate_stats


SCRIPT = '''
import pandas as pd
import numpy as np
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_eda import data_eda
from frontend.pages.data_viz import data_viz
StateManager.init_session()
if not st.session_state.get("initialized"):
    StateManager.load_dataset(pd.DataFrame({0: np.arange(12000, dtype=float), "y": np.arange(12000, dtype=float)*2, "category": pd.Series(["a", "b"]*6000, dtype="string")}), {})
    st.session_state.initialized = True
if st.button("Edit source", key="edit_source"):
    candidate = st.session_state.df.copy()
    candidate.iloc[0,0] = 12345
    StateManager.apply_operation("Edit source", candidate)
    st.rerun()
__PAGE__()
'''


class BoundedVisualizationTests(unittest.TestCase):
    def test_histogram_scatter_and_box_sample_deterministically_without_mutation(self):
        frame = pd.DataFrame({"x": np.arange(15000), "y": np.arange(15000) * 2})
        before = frame.copy(deep=True)
        histogram = viz.histogram(frame, "x")
        self.assertEqual(len(histogram.data[0].x), 10000)
        np.testing.assert_array_equal(histogram.data[0].x, viz.histogram(frame, "x").data[0].x)
        self.assertEqual(len(viz.scatter(frame, "x", "y").data[0].x), 10000)
        self.assertEqual(len(viz.box_plot(frame, "x").data[0].y), 10000)
        self.assertTrue(any("sample" in note for note in viz.chart_notes(histogram)))
        assert_frame_equal(frame, before)

    def test_bar_counts_and_aggregates_use_all_rows_before_top_n_selection(self):
        frame = pd.DataFrame({"group": [f"g{i}" for i in range(100)] * 200, "value": list(range(100)) * 200})
        counts = viz.bar_chart(frame, "group")
        self.assertEqual(len(counts.data[0].x), 50)
        self.assertTrue(all(value == 200 for value in counts.data[0].y))
        sums = viz.bar_chart(frame, "group", "value", "sum")
        self.assertEqual(len(sums.data[0].x), 50)
        self.assertEqual(sums.data[0].x[0], "g99")
        self.assertEqual(sums.data[0].y[0], 19800)

    def test_line_reduction_is_sorted_and_keeps_endpoints(self):
        frame = pd.DataFrame({"x": np.arange(15000)[::-1], "y": np.arange(15000)[::-1]})
        line = viz.line_chart(frame, "x", "y")
        self.assertEqual(len(line.data[0].x), 10000)
        self.assertEqual(line.data[0].x[0], 0)
        self.assertEqual(line.data[0].x[-1], 14999)
        self.assertTrue(np.all(np.diff(line.data[0].x) > 0))
        self.assertTrue(any("intermediate" in note for note in viz.chart_notes(line)))

    def test_category_color_groups_and_heatmap_are_bounded(self):
        frame = pd.DataFrame({"x": np.arange(2000), "y": np.arange(2000), "group": [f"c{i}" for i in range(100)] * 20})
        scatter = viz.scatter(frame, "x", "y", color="group")
        self.assertEqual(len(scatter.data), 50)
        self.assertTrue(any("categories" in note for note in viz.chart_notes(scatter)))
        matrix = pd.DataFrame(np.eye(100))
        self.assertEqual(np.asarray(viz.heatmap(matrix).data[0].z).shape, (50, 50))

    def test_date_line_preserves_full_time_range_before_reduction(self):
        dates = pd.date_range("2020-01-01", periods=15000, freq="h")
        frame = pd.DataFrame({"time": dates[::-1], "value": np.arange(15000)[::-1]})
        line = viz.line_chart(frame, "time", "value")
        self.assertEqual(len(line.data[0].x), 10000)
        self.assertEqual(pd.Timestamp(line.data[0].x[0]), dates[0])
        self.assertEqual(pd.Timestamp(line.data[0].x[-1]), dates[-1])
        self.assertFalse(any("categories" in note for note in viz.chart_notes(line)))

    def test_long_labels_cannot_bypass_chart_size_budget_or_enter_caches(self):
        frame = pd.DataFrame({"category": ["x" * 20000]})
        with patch("backend.visualization.MAX_CHART_BYTES", 10000):
            with self.assertRaisesRegex(ValueError, "size budget"):
                viz.bar_chart(frame, "category")
            script = SCRIPT.replace("__PAGE__", "data_eda").replace('0: np.arange(12000, dtype=float), "y": np.arange(12000, dtype=float)*2, "category": pd.Series(["a", "b"]*6000, dtype="string")', '"category": ["x"*20000]')
            app = AppTest.from_string(script).run(timeout=30)
        self.assertEqual([item.value for item in app.exception], [])
        self.assertTrue(any("size budget" in item.value for item in app.error))
        self.assertEqual(app.session_state["_analysis_cache"]["values"], {})

    def test_eda_executes_only_selected_tab_and_reuses_results_for_same_revision(self):
        with patch("frontend.pages.data_eda.univariate_stats", wraps=univariate_stats) as statistics, patch("frontend.pages.data_eda.correlation_matrix", wraps=correlation_matrix) as correlations:
            app = AppTest.from_string(SCRIPT.replace("__PAGE__", "data_eda")).run(timeout=30)
            self.assertEqual([item.value for item in app.exception], [])
            self.assertEqual(statistics.call_count, 1)
            self.assertEqual(correlations.call_count, 0)
            app.run()
            self.assertEqual(statistics.call_count, 1)
            app.session_state["eda_section"] = "Correlation"
            app.run()
            self.assertEqual([item.value for item in app.exception], [])
            self.assertEqual(correlations.call_count, 1)
            self.assertEqual(len(correlations.call_args.args[0]), 10000)
            app.run()
            self.assertEqual(correlations.call_count, 1)
            app.button(key="edit_source").click().run()
            self.assertEqual([item.value for item in app.exception], [])
            # Dataset invalidation returns to the initial tab and recalculates it.
            self.assertEqual(statistics.call_count, 2)

    def test_visualization_supports_zero_headers_string_columns_and_cache_invalidation(self):
        with patch("frontend.pages.data_viz.viz.histogram", wraps=viz.histogram) as histogram:
            app = AppTest.from_string(SCRIPT.replace("__PAGE__", "data_viz")).run(timeout=30)
            self.assertEqual([item.value for item in app.exception], [])
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertTrue(any("sample" in item.value for item in app.caption))
            self.assertEqual(histogram.call_count, 1)
            app.run()
            self.assertEqual(histogram.call_count, 1)
            app.button(key="edit_source").click().run()
            self.assertEqual(histogram.call_count, 2)
            app.selectbox(key="vz_type").set_value("Bar").run()
            self.assertIn("category", app.selectbox(key="vz_bar_x").options)
            self.assertEqual([item.value for item in app.exception], [])

    def test_failed_chart_save_is_actionable_and_does_not_claim_success(self):
        app = AppTest.from_string(SCRIPT.replace("__PAGE__", "data_viz")).run(timeout=30)
        with patch("frontend.pages.data_viz.StateManager.save_chart", side_effect=ValueError("Chart quota exceeded")):
            app.button(key="vz_save").click().run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertTrue(any("quota" in item.value for item in app.error))
        self.assertEqual(app.session_state["viz_charts"], [])


if __name__ == "__main__":
    unittest.main()
