import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.cleaning import (
    automatic_cleaning,
    build_missing_candidate,
    build_outlier_candidate,
    find_outliers,
    get_missing_strategies,
    get_missing_summary,
    get_outlier_columns,
    handle_missing,
    outlier_summary,
)


class CleaningPolicyTests(unittest.TestCase):
    def test_missingness_percentages_never_choose_destructive_defaults(self):
        frame = pd.DataFrame({
            "few": [None] + list(range(99)),
            "many": [None] * 90 + list(range(10)),
            "empty": [None] * 100,
        })
        before = frame.copy(deep=True)
        summary = get_missing_summary(frame)
        self.assertEqual(summary["suggested"].tolist(), ["keep", "keep", "keep"])
        self.assertIn("constant", summary.loc["empty", "guidance"])
        self.assertEqual(summary.loc["few", "%"], 1)
        self.assertEqual(summary.loc["many", "%"], 90)
        assert_frame_equal(frame, before)

    def test_guidance_considers_roles_and_categories(self):
        frame = pd.DataFrame({"id": [1, None], "code": [0, None]})
        summary = get_missing_summary(frame, {"id": {"role": "id"}, "code": {"is_categorical": True}})
        self.assertIn("role", summary.loc["id", "guidance"])
        self.assertIn("category", summary.loc["code", "guidance"])

    def test_empty_summary_and_keep_are_safe(self):
        empty = pd.DataFrame({"x": pd.Series([], dtype="Int64")})
        self.assertTrue(get_missing_summary(empty).empty)
        frame = pd.DataFrame({"x": pd.Series([1, None], dtype="Int64")})
        self.assertEqual(get_missing_strategies(frame["x"])[0], "keep")
        result = handle_missing(frame, "x", "keep")
        assert_frame_equal(result, frame)
        result.loc[0, "x"] = 9
        self.assertEqual(frame.loc[0, "x"], 1)

    def test_legacy_percentage_table_alone_does_not_remove_data(self):
        frame = pd.DataFrame({"x": [1, None, None]})
        result = automatic_cleaning(frame, pd.DataFrame({"%": [66.7]}, index=["x"]))
        assert_frame_equal(result, frame)
        self.assertIsNot(result, frame)

    def test_ids_ignored_columns_and_numeric_categories_require_opt_in(self):
        frame = pd.DataFrame({
            "measurement": [1.0, 2.0], "record_id": [1, 2], "code": [0, 1],
            "ignored": [1, 2], "timestamp_code": [1, 2], "flag": [True, False],
            "text": ["a", "b"], "complex": [1 + 2j, 3 + 4j],
        })
        metadata = {
            "record_id": {"role": "id"}, "code": {"is_categorical": True},
            "ignored": {"role": "ignore"}, "timestamp_code": {"role": "datetime"},
        }
        self.assertEqual(get_outlier_columns(frame, metadata), ["measurement"])
        self.assertEqual(
            get_outlier_columns(frame, metadata, include_protected=True),
            ["measurement", "record_id", "code", "ignored", "timestamp_code"],
        )


class MissingPlanTests(unittest.TestCase):
    def test_statistics_are_fitted_before_any_selected_rows_are_dropped(self):
        frame = pd.DataFrame({"dropper": [None, 1, 1], "value": [100.0, None, 2.0]})
        first = {"dropper": "drop_rows", "value": "mean"}
        second = dict(reversed(list(first.items())))
        result = build_missing_candidate(frame, first)
        self.assertEqual(result.index.tolist(), [1, 2])
        self.assertEqual(result.loc[1, "value"], 51)
        assert_frame_equal(result, build_missing_candidate(frame, second))

    def test_removal_counts_use_union_and_preserve_unrelated_duplicate_labels(self):
        frame = pd.DataFrame({"a": [None, 1, None, 1], "b": [1, None, None, 1]}, index=["same", "same", "other", "safe"])
        result = build_missing_candidate(frame, {"a": "drop_rows", "b": "drop_rows"})
        self.assertEqual(result.index.tolist(), ["safe"])
        duplicate = pd.DataFrame({"x": [None, 7, 8]}, index=["same", "same", "other"])
        kept = build_missing_candidate(duplicate, {"x": "drop_rows"})
        self.assertEqual(kept.index.tolist(), ["same", "other"])
        self.assertEqual(kept["x"].tolist(), [7, 8])

    def test_fills_keep_nullable_types_and_column_order(self):
        frame = pd.DataFrame({"a": pd.Series([1, 2, None], dtype="Int64"), "b": ["x", "y", None], "c": [1, 2, 3]})
        result = build_missing_candidate(frame, {"b": "constant", "a": "median", "c": "drop_column"}, {"b": "unknown"})
        self.assertEqual(result.columns.tolist(), ["a", "b"])
        self.assertEqual(str(result["a"].dtype), "Float64")
        self.assertEqual(result.loc[2, "a"], 1.5)
        self.assertEqual(result.loc[2, "b"], "unknown")

    def test_index_names_and_multiindex_survive_batch_transforms(self):
        index = pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2)], names=["group", "record"])
        frame = pd.DataFrame({"x": [None, 2.0, 4.0]}, index=index)
        result = build_missing_candidate(frame, {"x": "drop_rows"})
        pd.testing.assert_index_equal(result.index, index.take([1, 2]))

    def test_invalid_missing_batch_does_not_mutate_input(self):
        frame = pd.DataFrame({"a": pd.Series([1, None], dtype="Int64"), "b": pd.Series([2, None], dtype="Int64")})
        before = frame.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "b"):
            build_missing_candidate(frame, {"a": "constant", "b": "constant"}, {"a": "0", "b": "invalid"})
        assert_frame_equal(frame, before)
        for plan in ({"absent": "keep"}, {"a": "unknown"}):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                build_missing_candidate(frame, plan)
        assert_frame_equal(frame, before)


class OutlierPlanTests(unittest.TestCase):
    @staticmethod
    def frame():
        return pd.DataFrame({"a": list(range(9)) + [100], "b": list(range(8)) + [13, 100]})

    def test_all_bounds_use_same_base_before_removals(self):
        frame = self.frame()
        plan = {"a": ("iqr", "remove"), "b": ("iqr", "remove")}
        result = build_outlier_candidate(frame, plan)
        self.assertEqual(result.index.tolist(), list(range(9)))
        self.assertEqual(result.loc[8, "b"], 13)
        assert_frame_equal(result, build_outlier_candidate(frame, dict(reversed(list(plan.items())))))

    def test_cap_bounds_do_not_change_after_another_column_removes_rows(self):
        result = build_outlier_candidate(self.frame(), {"a": ("iqr", "remove"), "b": ("iqr", "cap")})
        self.assertEqual(result.loc[8, "b"], 13)

    def test_distinct_columns_remove_the_union_of_outlier_positions(self):
        frame = pd.DataFrame({"a": list(range(8)) + [100, 8], "b": list(range(9)) + [100]})
        result = build_outlier_candidate(frame, {"a": ("iqr", "remove"), "b": ("iqr", "remove")})
        self.assertEqual(result.index.tolist(), list(range(8)))

    def test_removing_outliers_preserves_valid_rows_with_duplicate_index_labels(self):
        frame = pd.DataFrame({"x": [0, 1, 2, 3, 100]}, index=["same", "b", "c", "d", "same"])
        result = build_outlier_candidate(frame, {"x": ("iqr", "remove")})
        self.assertEqual(result.index.tolist(), ["same", "b", "c", "d"])
        self.assertEqual(result.loc["same", "x"], 0)

    def test_ignore_does_not_fit_bounds_or_modify_nonfinite_values(self):
        frame = pd.DataFrame({"x": [1.0, np.inf, np.nan]})
        assert_frame_equal(build_outlier_candidate(frame, {"x": ("iqr", "ignore")}), frame)

    def test_selected_zscore_method_can_find_values_not_flagged_by_iqr(self):
        frame = pd.DataFrame({"x": [0] * 40 + [10] * 40 + [22]})
        self.assertEqual(find_outliers(frame, "iqr")["x"], [])
        self.assertEqual(find_outliers(frame, "zscore")["x"], [80])
        summary = outlier_summary(frame, {"x": "zscore"})
        self.assertEqual(summary.loc[0, "method"], "zscore")
        self.assertEqual(summary.loc[0, "outlier_count"], 1)

    def test_summary_bounds_preserve_large_integer_values(self):
        base = 2**53
        frame = pd.DataFrame({"x": pd.Series([base, base + 1, base + 2, base + 3, base + 100], dtype="Int64")})
        before = frame.copy(deep=True)
        summary = outlier_summary(frame, {"x": "iqr"})
        self.assertEqual(summary.columns.tolist(), ["column", "method", "outlier_count", "lower_bound", "upper_bound"])
        self.assertEqual(summary.loc[0, "upper_bound"], str(base + 6))
        self.assertEqual(summary.loc[0, "outlier_count"], 1)
        assert_frame_equal(frame, before)

    def test_empty_plans_return_owned_unchanged_frames(self):
        frame = self.frame()
        for builder in (build_missing_candidate, build_outlier_candidate):
            result = builder(frame, {})
            assert_frame_equal(result, frame)
            self.assertIsNot(result, frame)

    def test_invalid_outlier_batch_is_atomic(self):
        frame = self.frame()
        before = frame.copy(deep=True)
        for plan in ({"a": ("iqr", "cap"), "b": ("unknown", "remove")}, {"absent": ("iqr", "ignore")}):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                build_outlier_candidate(frame, plan)
        assert_frame_equal(frame, before)
        with self.assertRaises(ValueError):
            find_outliers(pd.DataFrame({"text": ["a", "b"]}), "unknown")


if __name__ == "__main__":
    unittest.main()
