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
from frontend.pages.data_upload import data_upload
from frontend.pages.data_cleaning import data_cleaning

StateManager.init_session()
if not st.session_state.get("test_initialized"):
    df = __FRAME__
    metadata = infer_column_meta(df)
    __RULES__
    StateManager.load_dataset(df, metadata)
    st.session_state.test_initialized = True

if st.button("Test undo", key="test_undo"):
    StateManager.undo()
    st.rerun()
__CONTROLS__
__PAGE__()
'''


class StateConsistencyPageTests(unittest.TestCase):
    def make_app(self, frame, *, rules="pass", page="data_upload", controls=""):
        source = SCRIPT.replace("__FRAME__", frame).replace("__RULES__", rules)
        source = source.replace("__CONTROLS__", controls).replace("__PAGE__", page)
        app = AppTest.from_string(source).run()
        self.assert_no_exceptions(app)
        return app

    def assert_no_exceptions(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    @staticmethod
    def button(app, label):
        return next(button for button in app.button if button.label == label)

    def test_schema_confirmation_is_undoable_through_upload_page(self):
        app = self.make_app('pd.DataFrame({"value": ["1", "2", "3"], "label": ["a", "b", "a"]})')
        original = app.session_state["df"].copy(deep=True)
        self.button(app, "Confirm Schema").click().run()
        self.assert_no_exceptions(app)
        self.assertTrue(app.session_state["schema_confirmed"])
        self.assertEqual(len(app.session_state["history"]), 1)
        self.assertTrue(pd.api.types.is_integer_dtype(app.session_state["df"]["value"]))
        app.button(key="test_undo").click().run()
        self.assert_no_exceptions(app)
        assert_frame_equal(app.session_state["df"], original)
        self.assertFalse(app.session_state["schema_confirmed"])
        self.assertIsNone(app.session_state["column_meta"]["value"]["user_dtype"])
        self.assertEqual(app.session_state["history"], [])

    def test_dropping_ruled_column_removes_metadata_before_mismatch_check(self):
        app = self.make_app(
            'pd.DataFrame({"value": pd.Series([1, None, 3], dtype="Int64"), "label": ["a", "b", "a"]})',
            rules='metadata["value"]["allowed_values"] = ["1", "3"]\n'
                  '    metadata["label"]["allowed_values"] = ["a", "b"]',
            page="data_cleaning",
        )
        app.selectbox(key="ms_value").set_value("drop_column")
        app.button(key="preview_missing").click().run()
        app.button(key="apply_missing").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(list(app.session_state["df"].columns), ["label"])
        self.assertEqual(list(app.session_state["column_meta"]), ["label"])
        app.button(key="find_mismatch").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("No mismatches" in success.value for success in app.success))

    def test_allowed_value_edits_require_apply_and_undo_restores_field(self):
        app = self.make_app(
            'pd.DataFrame({"label": ["a", "b", "a"]})',
            rules='metadata["label"]["allowed_values"] = ["a", "b"]',
            page="data_cleaning",
        )
        original = app.session_state["df"].copy(deep=True)
        app.text_input(key="av_label").set_value("").run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["column_meta"]["label"]["allowed_values"], ["a", "b"])
        self.assertEqual(app.session_state["history"], [])
        app.button(key="apply_allowed_values").click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["column_meta"]["label"]["allowed_values"])
        self.assertEqual(len(app.session_state["history"]), 1)
        assert_frame_equal(app.session_state["df"], original)
        app.button(key="test_undo").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["column_meta"]["label"]["allowed_values"], ["a", "b"])
        self.assertEqual(app.text_input(key="av_label").value, "a, b")
        app.run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["column_meta"]["label"]["allowed_values"], ["a", "b"])
        self.assertEqual(app.session_state["history"], [])

    def test_numeric_mismatch_check_survives_integer_to_float_cleaning(self):
        app = self.make_app(
            'pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})',
            rules='metadata["value"]["allowed_values"] = ["1", "2"]',
            page="data_cleaning",
        )
        app.selectbox(key="ms_value").set_value("median")
        app.button(key="preview_missing").click().run()
        app.button(key="apply_missing").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(str(app.session_state["df"]["value"].dtype), "Float64")
        app.button(key="find_mismatch").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.error), 0)
        preview = app.session_state["cleaning_previews"]["mismatch"]["report"]
        self.assertEqual(preview["before_sample"].index.tolist(), [2])
        self.assertEqual(preview["before_sample"]["value"].tolist(), [1.5])
        self.assertEqual(preview["summary"]["rows_removed"], 1)

    def test_invalid_numeric_rule_is_actionable_and_atomic_across_buttons(self):
        for action in ("find_mismatch", "apply_allowed_values", "apply_mismatch"):
            with self.subTest(action=action):
                app = self.make_app(
                    'pd.DataFrame({"value": pd.Series([1, 2, 3], dtype="Int64")})',
                    rules='metadata["value"]["allowed_values"] = ["1", "2", "3"]',
                    page="data_cleaning",
                )
                original = app.session_state["df"].copy(deep=True)
                metadata = copy.deepcopy(app.session_state["column_meta"])
                app.text_input(key="av_value").set_value("5 out of 10").run()
                app.session_state["viz_charts"] = ["saved chart"]
                app.button(key="find_mismatch" if action == "apply_mismatch" else action).click().run()
                self.assert_no_exceptions(app)
                self.assertEqual(len(app.error), 1)
                self.assertIn("value", app.error[0].value)
                if action == "apply_mismatch":
                    self.assertTrue(app.button(key="apply_mismatch").disabled)
                assert_frame_equal(app.session_state["df"], original)
                self.assertEqual(app.session_state["column_meta"], metadata)
                self.assertEqual(app.session_state["history"], [])
                self.assertEqual(app.session_state["viz_charts"], ["saved chart"])

    def test_small_and_empty_uploads_remain_clearable(self):
        for expression in (
            'pd.DataFrame({"value": ["1"]})',
            'pd.DataFrame({"value": pd.Series([], dtype="Int64")})',
            'pd.DataFrame(index=range(3))',
        ):
            with self.subTest(expression=expression):
                app = self.make_app(expression)
                self.button(app, "Clear Dataset").click().run()
                self.assert_no_exceptions(app)
                self.assertIsNone(app.session_state["df"])
                self.assertEqual(app.session_state["history"], [])
                self.assertEqual(app.session_state["viz_charts"], [])

    def test_empty_operation_results_can_be_undone_on_upload_page(self):
        for operation in ("st.session_state.df.iloc[:0].copy()", "st.session_state.df.iloc[:, :0].copy()"):
            with self.subTest(operation=operation):
                controls = (
                    'if st.button("Empty working data", key="test_empty"):\n'
                    f'    StateManager.apply_operation("Empty working data", {operation})\n'
                    '    st.rerun()'
                )
                app = self.make_app('pd.DataFrame({"value": ["1", "2", "3"]})', controls=controls)
                original = app.session_state["df"].copy(deep=True)
                app.button(key="test_empty").click().run()
                self.assert_no_exceptions(app)
                self.assertTrue(app.session_state["df"].empty)
                self.assertEqual(len(app.session_state["history"]), 1)
                app.button(key="test_undo").click().run()
                self.assert_no_exceptions(app)
                assert_frame_equal(app.session_state["df"], original)
                self.assertEqual(app.session_state["history"], [])

    def test_replacement_and_clear_reset_active_widgets_and_saved_charts(self):
        controls = '''
if StateManager.has_dataset():
    st.selectbox("Remembered target", st.session_state.df.columns.tolist(), key="ml_target")
if st.button("Replace working data", key="test_replace"):
    df = pd.DataFrame({"different": [4, 5, 6]})
    StateManager.load_dataset(df, infer_column_meta(df))
    st.rerun()
'''
        app = self.make_app('pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "a"]})', controls=controls)
        app.selectbox(key="ml_target").set_value("label").run()
        self.assert_no_exceptions(app)
        app.session_state["viz_charts"] = ["old chart"]
        app.session_state["ui_theme"] = "dark"
        app.button(key="test_replace").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.selectbox(key="ml_target").value, "different")
        self.assertEqual(app.session_state["viz_charts"], [])
        self.assertEqual(app.session_state["ui_theme"], "dark")
        app.session_state["viz_charts"] = ["replacement chart"]
        self.button(app, "Clear Dataset").click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["df"])
        self.assertEqual(app.session_state["viz_charts"], [])
        self.assertEqual(app.session_state["ui_theme"], "dark")
        self.assertNotIn("ml_target", app.session_state)


if __name__ == "__main__":
    unittest.main()
