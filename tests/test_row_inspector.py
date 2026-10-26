import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from frontend.components.row_inspector import filter_rows


SCRIPT = '''
import pandas as pd
import streamlit as st
from frontend.components.row_inspector import row_inspector
if "frame" not in st.session_state:
    st.session_state.frame = __FRAME__
row_inspector(st.session_state.frame)
'''


class RowInspectorTests(unittest.TestCase):
    def test_search_is_literal_case_insensitive_and_keeps_duplicate_indices(self):
        frame = pd.DataFrame({"text": ["a.b", "A.B", "axb", None]}, index=[7, 7, 9, 9])
        before = frame.copy(deep=True)
        result = filter_rows(frame, ["text"], "A.B")
        assert_frame_equal(result, frame.iloc[:2])
        assert_frame_equal(frame, before)

    def test_selected_columns_and_missing_filter_are_combined(self):
        frame = pd.DataFrame({"a": ["yes", "yes", "no"], "b": [None, "yes", None]})
        result = filter_rows(frame, ["a", "b"], "yes", True)
        assert_frame_equal(result, frame.iloc[[0]])
        self.assertTrue(filter_rows(frame, ["a"], "", True).empty)

    def test_integer_tuple_headers_and_non_scalar_objects_are_searchable(self):
        frame = pd.DataFrame({0: [{"key": "Found"}, ["other"], pd.NA], ("x", "y"): [1, 2, 3]})
        result = filter_rows(frame, [0, ("x", "y")], "found")
        assert_frame_equal(result, frame.iloc[[0]])
        self.assertEqual(result.columns.tolist(), [0, ("x", "y")])

    def page(self, expression):
        app = AppTest.from_string(SCRIPT.replace("__FRAME__", expression)).run(timeout=30)
        self.assertEqual([item.value for item in app.exception], [])
        return app

    def test_ui_search_column_selection_and_page_are_read_only(self):
        app = self.page('pd.DataFrame({"text": ["A.B"] * 50 + ["other"] * 50, "number": list(range(100))}, index=[8] * 100)')
        original = app.session_state["frame"].copy(deep=True)
        self.assertEqual(len(app.dataframe[0].value), 25)
        app.number_input(key="inspect_page").set_value(3).run()
        self.assertEqual(app.dataframe[0].value["number"].iloc[0], 50)
        app.text_input(key="inspect_query").set_value("a.b").run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual(app.number_input(key="inspect_page").value, 1)
        self.assertEqual(len(app.dataframe[0].value), 25)
        app.multiselect(key="inspect_columns").set_value(["text"]).run()
        self.assertEqual(app.dataframe[0].value.columns.tolist(), ["text"])
        assert_frame_equal(app.session_state["frame"], original)

    def test_ui_handles_no_columns_no_rows_and_no_matches(self):
        for expression in ('pd.DataFrame()', 'pd.DataFrame({"a": pd.Series([], dtype="string")})'):
            with self.subTest(expression=expression):
                self.page(expression)
        app = self.page('pd.DataFrame({"a": [None, None]})')
        app.text_input(key="inspect_query").set_value("missing").run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertTrue(any("No rows match" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
