import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.cleaning_preview import describe_cleaning_preview


class CleaningPreviewTests(unittest.TestCase):
    def test_mixed_changes_have_exact_totals_without_mutating_inputs(self):
        before = pd.DataFrame({
            "number": pd.Series([1, None, 3, 4], dtype="Int64"),
            "label": pd.Series(["a", "b", None, "d"], dtype="string"),
        })
        after = before.copy()
        after.loc[1, "number"] = 2
        after.loc[2, "label"] = "filled"
        after.loc[0, "number"] = pd.NA
        after = after.iloc[:3]
        original_before, original_after = before.copy(), after.copy()
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["rows_removed"], 1)
        self.assertEqual(result["summary"]["changed_rows"], 3)
        self.assertEqual(result["summary"]["changed_cells"], 3)
        self.assertEqual(result["summary"]["filled_missing"], 2)
        self.assertEqual(result["summary"]["introduced_missing"], 1)
        self.assertEqual(result["affected_positions"].tolist(), [0, 1, 2, 3])
        assert_frame_equal(before, original_before)
        assert_frame_equal(after, original_after)
        result["candidate"].loc[1, "number"] = 99
        assert_frame_equal(after, original_after)

    def test_all_kinds_of_missing_values_compare_equal(self):
        before = pd.DataFrame({
            "number": pd.Series([None, 1], dtype="Int64"),
            "date": pd.to_datetime([None, "2024-01-01"]),
            "object": pd.Series([None, "a"], dtype=object),
        })
        after = before.copy()
        after.loc[0, "object"] = np.nan
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["changed_cells"], 0)
        self.assertEqual(result["summary"]["changed_rows"], 0)
        self.assertTrue(result["column_changes"].empty)
        self.assertTrue(result["before_sample"].empty)

    def test_numeric_equal_values_only_report_dtype_change(self):
        before = pd.DataFrame({"x": pd.Series([1, 2, None], dtype="Int64")})
        after = before.astype({"x": "Float64"})
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["changed_cells"], 0)
        self.assertEqual(result["column_changes"].iloc[0]["before_dtype"], "Int64")
        self.assertEqual(result["column_changes"].iloc[0]["after_dtype"], "Float64")
        self.assertTrue(result["before_sample"].empty)

    def test_comparison_does_not_round_large_integers(self):
        before = pd.DataFrame({"x": pd.Series([2**53 + 1, 2**53 + 2], dtype="Int64")})
        after = pd.DataFrame({"x": pd.Series([float(2**53), float(2**53 + 2)], dtype="Float64")})
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["changed_cells"], 1)
        self.assertEqual(result["affected_positions"].tolist(), [0])

    def test_different_category_sets_can_be_compared(self):
        before = pd.DataFrame({"x": pd.Categorical(["a", None, "b"])})
        after = pd.DataFrame({"x": pd.Categorical(["a", "new", "b"], categories=["a", "b", "new"])})
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["changed_cells"], 1)
        self.assertEqual(result["summary"]["filled_missing"], 1)
        self.assertEqual(result["after_sample"]["x"].tolist(), ["new"])

    def test_removed_rows_and_modifications_form_a_positional_union(self):
        before = pd.DataFrame({"x": [10, 20, 30, 40, 50]}, index=["same", "same", "other", "same", "last"])
        after = before.reset_index(drop=True).iloc[[0, 2, 4]].copy()
        after.loc[2, "x"] = 31
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["rows_removed"], 2)
        self.assertEqual(result["summary"]["changed_rows"], 1)
        self.assertEqual(result["affected_positions"].tolist(), [1, 2, 3])
        self.assertEqual(result["candidate"].index.tolist(), ["same", "other", "last"])
        self.assertEqual(result["before_sample"].index.tolist(), ["same", "other", "same"])
        self.assertEqual(result["after_sample"].index.tolist(), ["other"])
        self.assertEqual(result["after_sample"]["x"].tolist(), [31])

    def test_dropped_columns_are_separate_from_retained_cell_changes(self):
        before = pd.DataFrame({"keep": [1, 2, 3], "drop": [None, "b", "c"]}, index=[8, 9, 10])
        after = before.reset_index(drop=True).drop(columns="drop")
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["dropped_columns"], ["drop"])
        self.assertEqual(result["summary"]["columns_removed"], 1)
        self.assertEqual(result["summary"]["changed_cells"], 0)
        self.assertEqual(result["summary"]["changed_rows"], 0)
        self.assertEqual(result["affected_positions"].tolist(), [0, 1, 2])
        self.assertEqual(result["before_sample"].index.tolist(), [8, 9, 10])
        self.assertEqual(result["after_sample"].index.tolist(), [8, 9, 10])

    def test_samples_are_bounded_and_correspond_to_the_same_original_rows(self):
        before = pd.DataFrame({"x": range(100)}, index=range(1000, 1100))
        after = before.reset_index(drop=True).iloc[::2].copy()
        after["x"] += 100
        result = describe_cleaning_preview(before, after, sample_size=7)
        self.assertEqual(len(result["before_sample"]), 7)
        self.assertEqual(result["after_sample"].index.tolist(), [1000, 1002, 1004, 1006])
        self.assertTrue(set(result["after_sample"].index) <= set(result["before_sample"].index))

    def test_empty_frames_and_zero_sample_size_are_valid(self):
        before = pd.DataFrame({"x": pd.Series([], dtype="Int64")})
        result = describe_cleaning_preview(before, before.copy(), sample_size=0)
        self.assertEqual(result["summary"]["rows_before"], 0)
        self.assertEqual(result["summary"]["changed_cells"], 0)
        self.assertTrue(result["before_sample"].empty)
        before = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        result = describe_cleaning_preview(before, before.reset_index(drop=True).iloc[:0])
        self.assertEqual(result["summary"]["rows_removed"], 2)
        self.assertEqual(result["before_sample"].index.tolist(), ["a", "b"])
        self.assertTrue(result["after_sample"].empty)

    def test_dropping_all_columns_preserves_remaining_row_identity(self):
        before = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["same", "same"], name="row"))
        after = pd.DataFrame(index=pd.Index([1]))
        result = describe_cleaning_preview(before, after)
        self.assertEqual(result["summary"]["rows_after"], 1)
        self.assertEqual(result["summary"]["columns_removed"], 1)
        self.assertEqual(result["candidate"].index.tolist(), ["same"])
        self.assertEqual(result["candidate"].index.name, "row")

    def test_invalid_position_mappings_are_rejected(self):
        before = pd.DataFrame({"x": [1, 2, 3]})
        for index in ([0, 0], [2, 1], [-1], [3], [0.0], [False], ["0"]):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "position|order"):
                describe_cleaning_preview(before, pd.DataFrame({"x": [1] * len(index)}, index=index))

    def test_new_or_reordered_columns_are_rejected(self):
        before = pd.DataFrame({"a": [1], "b": [2]})
        for after in (pd.DataFrame({"new": [1]}), before[["b", "a"]], pd.DataFrame([[1, 2]], columns=["a", "a"])):
            with self.subTest(columns=after.columns.tolist()), self.assertRaisesRegex(ValueError, "column"):
                describe_cleaning_preview(before, after)

    def test_invalid_sample_sizes_are_rejected(self):
        before = pd.DataFrame({"x": [1]})
        for size in (-1, 1.5, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "sample size"):
                describe_cleaning_preview(before, before.copy(), size)


if __name__ == "__main__":
    unittest.main()
