import copy
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest


SCRIPT = '''
import pandas as pd
import streamlit as st
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_cleaning import data_cleaning

StateManager.init_session()
if not st.session_state.get("test_initialized"):
    df = __FRAME__
    StateManager.load_dataset(df, infer_column_meta(df))
    st.session_state.viz_charts = ["saved chart"]
    st.session_state.test_initialized = True
if st.button("Test external edit", key="test_edit"):
    candidate = st.session_state.df.copy()
    candidate.iloc[0, 0] = 100
    StateManager.apply_operation("External edit", candidate)
    st.rerun()
if st.button("Test undo", key="test_undo"):
    StateManager.undo()
    st.rerun()
data_cleaning()
'''


class CleaningPolicyPageTests(unittest.TestCase):
    def make_app(self, frame):
        app = AppTest.from_string(SCRIPT.replace("__FRAME__", frame)).run(timeout=30)
        self.assert_no_exceptions(app)
        return app

    def assert_no_exceptions(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def assert_preview_did_not_apply(self, app, frame, metadata):
        self.assert_no_exceptions(app)
        assert_frame_equal(app.session_state["df"], frame)
        self.assertEqual(app.session_state["column_meta"], metadata)
        self.assertEqual(app.session_state["history"], [])
        self.assertEqual(app.session_state["viz_charts"], ["saved chart"])

    def select_outlier_columns(self, app, columns):
        app.checkbox(key="outlier_include_protected").set_value(True).run()
        self.assert_no_exceptions(app)
        app.multiselect(key="outlier_columns").set_value(columns).run()
        self.assert_no_exceptions(app)

    def test_keep_and_ignore_are_safe_defaults(self):
        app = self.make_app('pd.DataFrame({"value": [float(i) for i in range(40)] + [1000.0, None]})')
        self.assertEqual(app.selectbox(key="ms_value").value, "keep")
        self.assertEqual(app.selectbox(key="oa_value").value, "ignore")
        self.assertTrue(app.button(key="apply_missing").disabled)
        self.assertTrue(app.button(key="apply_outliers").disabled)
        original = app.session_state["df"].copy(deep=True)
        metadata = copy.deepcopy(app.session_state["column_meta"])
        app.button(key="preview_missing").click().run()
        self.assert_preview_did_not_apply(app, original, metadata)
        self.assertTrue(app.button(key="apply_missing").disabled)

    def test_missing_preview_is_read_only_then_applies_exact_candidate(self):
        app = self.make_app('pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})')
        original = app.session_state["df"].copy(deep=True)
        metadata = copy.deepcopy(app.session_state["column_meta"])
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        self.assert_preview_did_not_apply(app, original, metadata)
        self.assertFalse(app.button(key="apply_missing").disabled)
        candidate = app.session_state["cleaning_previews"]["missing"]["report"]["candidate"].copy(deep=True)
        self.assertEqual(candidate.loc[2, "value"], 1.5)
        app.button(key="apply_missing").click().run()
        self.assert_no_exceptions(app)
        assert_frame_equal(app.session_state["df"], candidate)
        self.assertEqual(len(app.session_state["history"]), 1)
        assert_frame_equal(app.session_state["history"][-1]["df"], original)
        self.assertEqual(app.session_state["viz_charts"], [])
        self.assertEqual(app.session_state["cleaning_previews"], {})

    def test_changing_controls_invalidates_a_previous_preview(self):
        app = self.make_app('pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})')
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        self.assertFalse(app.button(key="apply_missing").disabled)
        app.selectbox(key="ms_value").set_value("constant").run()
        self.assert_no_exceptions(app)
        self.assertTrue(app.button(key="apply_missing").disabled)
        app.text_input(key="fv_value").set_value("0").run()
        self.assertTrue(app.button(key="apply_missing").disabled)
        app.button(key="preview_missing").click().run()
        self.assert_no_exceptions(app)
        app.button(key="apply_missing").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1, 2, 0])

    def test_dataset_edits_and_undo_invalidate_pending_previews(self):
        app = self.make_app('pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})')
        original = app.session_state["df"].copy(deep=True)
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        app.button(key="test_edit").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["cleaning_previews"], {})
        self.assertTrue(app.button(key="apply_missing").disabled)
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        app.button(key="test_undo").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["cleaning_previews"], {})
        self.assertTrue(app.button(key="apply_missing").disabled)
        assert_frame_equal(app.session_state["df"], original)

    def test_numeric_ids_and_codes_require_explicit_outlier_opt_in(self):
        app = self.make_app(
            'pd.DataFrame({"measurement": [float(i) for i in range(40)], '
            '"record_id": list(range(40)), "code": [0, 1] * 20})'
        )
        self.assertFalse(app.checkbox(key="outlier_include_protected").value)
        self.assertEqual(app.multiselect(key="outlier_columns").options, ["measurement"])
        self.select_outlier_columns(app, ["record_id", "code"])
        self.assertEqual(app.selectbox(key="oa_record_id").value, "ignore")
        self.assertEqual(app.selectbox(key="oa_code").value, "ignore")

    def test_zscore_controls_work_without_any_iqr_outliers(self):
        app = self.make_app('pd.DataFrame({"value": [0] * 40 + [10] * 40 + [22]})')
        self.select_outlier_columns(app, ["value"])
        self.assertEqual(app.selectbox(key="om_value").value, "iqr")
        app.selectbox(key="om_value").set_value("zscore").run()
        self.assert_no_exceptions(app)
        summaries = [table.value for table in app.dataframe if "outlier_count" in table.value.columns]
        self.assertTrue(any(table["outlier_count"].tolist() == [1] for table in summaries))
        app.selectbox(key="oa_value").set_value("remove")
        app.button(key="preview_outliers").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.session_state["df"]), 81)
        self.assertEqual(app.session_state["cleaning_previews"]["outliers"]["report"]["summary"]["rows_removed"], 1)
        app.button(key="apply_outliers").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.session_state["df"]), 80)
        self.assertNotIn(22, app.session_state["df"]["value"].tolist())

    def test_duplicate_preview_totals_follow_the_selected_keep_policy(self):
        for keep, removed in (("first", 2), (False, 4)):
            with self.subTest(keep=keep):
                app = self.make_app('pd.DataFrame({"value": ["a", "a", "b", "c", "c", "d"]})')
                original = app.session_state["df"].copy(deep=True)
                metadata = copy.deepcopy(app.session_state["column_meta"])
                app.radio(key="dup_keep").set_value(keep)
                app.button(key="preview_dupes").click().run()
                self.assert_preview_did_not_apply(app, original, metadata)
                report = app.session_state["cleaning_previews"]["dupes"]["report"]
                self.assertEqual(report["summary"]["rows_removed"], removed)
                candidate = report["candidate"].copy(deep=True)
                assert_frame_equal(candidate, original.drop_duplicates(keep=keep))
                app.button(key="apply_dupes").click().run()
                self.assert_no_exceptions(app)
                assert_frame_equal(app.session_state["df"], candidate)

    def test_missing_removal_preview_counts_cumulative_row_union(self):
        app = self.make_app('pd.DataFrame({"a": [None, 1, None, 1], "b": [1, None, None, 1]})')
        app.selectbox(key="ms_a").set_value("drop_rows")
        app.selectbox(key="ms_b").set_value("drop_rows")
        app.button(key="preview_missing").click().run()
        self.assert_no_exceptions(app)
        report = app.session_state["cleaning_previews"]["missing"]["report"]
        self.assertEqual(report["summary"]["rows_removed"], 3)
        self.assertEqual(report["candidate"].index.tolist(), [3])
        self.assertEqual(len(app.session_state["df"]), 4)
        app.button(key="apply_missing").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["df"].index.tolist(), [3])

    def test_mismatch_preview_preserves_valid_rows_sharing_an_index_label(self):
        app = self.make_app('pd.DataFrame({"value": [1, 2, 1]}, index=["same", "same", "other"])')
        original = app.session_state["df"].copy(deep=True)
        metadata = copy.deepcopy(app.session_state["column_meta"])
        app.text_input(key="av_value").set_value("1")
        app.button(key="find_mismatch").click().run()
        self.assert_preview_did_not_apply(app, original, metadata)
        report = app.session_state["cleaning_previews"]["mismatch"]["report"]
        self.assertEqual(report["summary"]["rows_removed"], 1)
        self.assertEqual(report["candidate"].index.tolist(), ["same", "other"])
        app.button(key="apply_mismatch").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["df"].index.tolist(), ["same", "other"])
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
