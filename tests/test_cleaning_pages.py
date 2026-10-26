import copy
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest


CLEANING_SCRIPT = '''
import pandas as pd
from backend.state import StateManager
from frontend.pages.data_cleaning import data_cleaning

StateManager.init_session()
if not StateManager.has_dataset():
    df = __FRAME__
    StateManager.load_dataset(df, {})
    StateManager.save_snapshot("Initial checkpoint")
data_cleaning()
'''


class CleaningPageTests(unittest.TestCase):
    def make_app(self, frame_expression):
        app = AppTest.from_string(CLEANING_SCRIPT.replace("__FRAME__", frame_expression)).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        return app

    def assert_history_unchanged(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for current, previous in zip(actual, expected):
            self.assertEqual(current["name"], previous["name"])
            self.assertEqual(current["column_meta"], previous["column_meta"])
            assert_frame_equal(current["df"], previous["df"])

    def test_strategy_controls_offer_only_compatible_operations(self):
        app = self.make_app(
            'pd.DataFrame({"number": pd.Series([1, 2, None], dtype="Int64"), '
            '"text": pd.Series(["a", "b", None], dtype="string"), '
            '"flag": pd.Series([True, False, None], dtype="boolean"), '
            '"date": pd.to_datetime(["2024-01-01", "2024-02-01", None])})'
        )
        numeric = {"mean", "median", "interpolate"}
        self.assertTrue(numeric <= set(app.selectbox(key="ms_number").options))
        for column in ("text", "flag", "date"):
            with self.subTest(column=column):
                options = set(app.selectbox(key=f"ms_{column}").options)
                self.assertFalse(numeric & options)
                self.assertTrue({"mode", "constant", "drop_rows", "drop_column"} <= options)

    def test_nullable_integer_median_fill_succeeds_and_records_history(self):
        app = self.make_app('pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})')
        original = app.session_state["df"].copy(deep=True)
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        app.button(key="apply_missing").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1.0, 2.0, 1.5])
        self.assertEqual(str(app.session_state["df"]["value"].dtype), "Float64")
        self.assertEqual(len(app.session_state["history"]), 2)
        assert_frame_equal(app.session_state["history"][-1]["df"], original)

    def test_nullable_integer_fractional_cap_succeeds_and_retains_missing_row(self):
        app = self.make_app('pd.DataFrame({"value": pd.Series([0, 1, 2, 4, 100, None], dtype="Int64")})')
        original = app.session_state["df"].copy(deep=True)
        app.checkbox(key="outlier_include_protected").check().run()
        app.multiselect(key="outlier_columns").set_value(["value"]).run()
        app.selectbox(key="oa_value").set_value("cap")
        app.button(key="preview_outliers").click().run()
        app.button(key="apply_outliers").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        result = app.session_state["df"]
        self.assertEqual(result.loc[4, "value"], 8.5)
        self.assertEqual(str(result["value"].dtype), "Float64")
        self.assertTrue(pd.isna(result.loc[5, "value"]))
        self.assertEqual(len(result), 6)
        self.assertEqual(len(app.session_state["history"]), 2)
        assert_frame_equal(app.session_state["history"][-1]["df"], original)

    def test_invalid_constant_leaves_entire_missing_value_batch_unchanged(self):
        app = self.make_app(
            'pd.DataFrame({"first": pd.Series([1, 2, None], dtype="Int64"), '
            '"second": pd.Series([3, 4, None], dtype="Int64")})'
        )
        original = app.session_state["df"].copy(deep=True)
        history = copy.deepcopy(app.session_state["history"])
        app.selectbox(key="ms_first").set_value("constant")
        app.selectbox(key="ms_second").set_value("constant")
        app.run()
        app.text_input(key="fv_first").set_value("0")
        app.text_input(key="fv_second").set_value("invalid number")
        app.button(key="preview_missing").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("second", app.error[0].value)
        self.assertTrue(app.button(key="apply_missing").disabled)
        assert_frame_equal(app.session_state["df"], original)
        self.assert_history_unchanged(app.session_state["history"], history)

    def test_unsafe_cap_leaves_entire_outlier_batch_unchanged(self):
        app = self.make_app(
            'pd.DataFrame({"first": pd.Series([0, 1, 2, 4, 100, None], dtype="Int64"), '
            '"second": pd.Series([2**53, 2**53 + 1, 2**53 + 2, 2**53 + 4, 2**53 + 100, None], dtype="Int64")})'
        )
        original = app.session_state["df"].copy(deep=True)
        history = copy.deepcopy(app.session_state["history"])
        app.checkbox(key="outlier_include_protected").check().run()
        app.multiselect(key="outlier_columns").set_value(["first", "second"]).run()
        app.selectbox(key="oa_first").set_value("cap")
        app.selectbox(key="oa_second").set_value("cap")
        app.button(key="preview_outliers").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("second", app.error[0].value)
        self.assertTrue(app.button(key="apply_outliers").disabled)
        assert_frame_equal(app.session_state["df"], original)
        self.assert_history_unchanged(app.session_state["history"], history)


if __name__ == "__main__":
    unittest.main()
