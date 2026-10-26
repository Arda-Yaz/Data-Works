import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.eda import correlation_matrix, group_by_summary, univariate_stats, value_counts_summary
from backend.visualization import bar_chart, histogram


class UnivariateTests(unittest.TestCase):
    def test_all_missing_nullable_numbers_have_json_null_statistics(self):
        for dtype in ("Int64", "Float64"):
            with self.subTest(dtype=dtype):
                result = univariate_stats(pd.DataFrame({"value": pd.Series([None, None], dtype=dtype)}), "value")
                self.assertEqual(result["type"], "numeric")
                self.assertEqual(result["count"], 0)
                self.assertEqual(result["finite_count"], 0)
                for field in ("mean", "median", "std", "min", "max", "q25", "q75", "skew", "kurtosis"):
                    self.assertIsNone(result[field])
                json.dumps(result, allow_nan=False)

    def test_single_number_reports_only_defined_statistics(self):
        result = univariate_stats(pd.DataFrame({"value": pd.Series([7, None], dtype="Int64")}), "value")
        self.assertEqual(result["mean"], 7)
        self.assertEqual(result["median"], 7)
        self.assertIsNone(result["std"])
        self.assertIsNone(result["skew"])
        self.assertIsNone(result["kurtosis"])
        json.dumps(result, allow_nan=False)

    def test_nonfinite_observations_are_counted_and_explicitly_excluded(self):
        result = univariate_stats(pd.DataFrame({"value": [1.0, 2.0, np.inf, -np.inf, None]}), "value")
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["finite_count"], 2)
        self.assertEqual(result["non_finite"], 2)
        self.assertEqual(result["mean"], 1.5)
        json.dumps(result, allow_nan=False)

    def test_all_nonfinite_and_extreme_values_never_emit_invalid_json(self):
        for values in ([np.inf, -np.inf], [1e308, -1e308, 1e308, -1e308]):
            with self.subTest(values=values):
                result = univariate_stats(pd.DataFrame({"value": values}), "value")
                json.dumps(result, allow_nan=False)

    def test_large_integer_extrema_do_not_round(self):
        value = 2**53 + 1
        result = univariate_stats(pd.DataFrame({"value": pd.Series([value], dtype="Int64")}), "value")
        self.assertEqual(result["min"], value)
        self.assertEqual(result["max"], value)

    def test_empty_categorical_column_has_no_phantom_mode_or_counts(self):
        frame = pd.DataFrame({"label": pd.Categorical([None, None], categories=["unused", "also unused"])})
        result = univariate_stats(frame, "label")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["unique"], 0)
        self.assertIsNone(result["mode"])
        self.assertEqual(result["top_5"], {})

    def test_boolean_data_is_categorical_and_json_serializable(self):
        result = univariate_stats(pd.DataFrame({"flag": pd.Series([True, False, None], dtype="boolean")}), "flag")
        self.assertEqual(result["type"], "categorical")
        self.assertEqual(result["unique"], 2)
        json.dumps(result, allow_nan=False)

    def test_categorical_json_keys_preserve_distinct_display_collisions(self):
        result = univariate_stats(pd.DataFrame({"label": pd.Series([1, "1", 1], dtype=object)}), "label")
        self.assertEqual(len(result["top_5"]), 2)
        self.assertEqual(sum(result["top_5"].values()), 3)
        json.dumps(result, allow_nan=False)


class SummaryTests(unittest.TestCase):
    def test_grouping_and_aggregating_the_same_column(self):
        frame = pd.DataFrame({"x": [1, 1, 2, 2]})
        before = frame.copy(deep=True)
        result = group_by_summary(frame, "x", "x", "sum")
        self.assertEqual(result.columns.tolist(), ["x", "sum(x)"])
        self.assertEqual(result.iloc[:, 0].tolist(), [1, 2])
        self.assertEqual(result.iloc[:, 1].tolist(), [2, 4])
        assert_frame_equal(frame, before)

    def test_generated_aggregate_name_cannot_overwrite_group_name(self):
        frame = pd.DataFrame({"mean(x)": ["a", "a", "b"], "x": [1.0, 3.0, 9.0]})
        result = group_by_summary(frame, "mean(x)", "x")
        self.assertTrue(result.columns.is_unique)
        self.assertEqual(result.iloc[:, 0].tolist(), ["a", "b"])
        self.assertEqual(result.iloc[:, 1].tolist(), [2.0, 9.0])

    def test_count_named_column_has_distinct_count_output(self):
        result = value_counts_summary(pd.DataFrame({"count": ["a", "a", "b"]}), "count")
        self.assertTrue(result.columns.is_unique)
        self.assertEqual(result.iloc[:, 0].tolist(), ["a", "b"])
        self.assertEqual(result.iloc[:, 1].tolist(), [2, 1])

    def test_unused_category_levels_do_not_appear_as_groups_or_counts(self):
        frame = pd.DataFrame({"label": pd.Categorical(["a", "a"], categories=["a", "unused"]), "value": [1, 3]})
        counts = value_counts_summary(frame, "label")
        groups = group_by_summary(frame, "label", "value")
        self.assertEqual(counts.iloc[:, 0].tolist(), ["a"])
        self.assertEqual(groups.iloc[:, 0].tolist(), ["a"])
        self.assertEqual(groups.iloc[:, 1].tolist(), [2.0])

    def test_sum_of_all_missing_observations_remains_missing(self):
        frame = pd.DataFrame({"group": ["a", "b"], "value": pd.Series([None, 1], dtype="Int64")})
        result = group_by_summary(frame, "group", "value", "sum")
        self.assertTrue(pd.isna(result.iloc[0, 1]))
        self.assertEqual(result.iloc[1, 1], 1)

    def test_numeric_header_labels_are_supported(self):
        frame = pd.DataFrame({0: [1, 1, 2], 5: [2.0, 4.0, 9.0]})
        result = group_by_summary(frame, 0, 5)
        self.assertEqual(result.columns.tolist(), [0, "mean(5)"])
        self.assertEqual(result.iloc[:, 1].tolist(), [3.0, 9.0])

    def test_empty_summaries_keep_well_defined_columns(self):
        frame = pd.DataFrame({"x": pd.Series([], dtype="Int64")})
        self.assertEqual(group_by_summary(frame, "x", "x").columns.tolist(), ["x", "mean(x)"])
        self.assertEqual(value_counts_summary(frame, "x").columns.tolist(), ["x", "count"])

    def test_unsupported_functions_columns_and_dtypes_are_actionable(self):
        frame = pd.DataFrame({"label": ["a", "b"], "value": [1, 2]})
        for args in (("label", "value", "unsupported"), ("label", "label", "mean"), ("missing", "value", "sum")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                group_by_summary(frame, *args)

    def test_constant_and_empty_columns_are_excluded_from_correlation(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0], "constant": [1, 1, 1], "empty": pd.Series([None] * 3, dtype="Int64")})
        result = correlation_matrix(frame)
        self.assertEqual(result.columns.tolist(), ["x", "y"])
        self.assertEqual(result.loc["x", "y"], 1.0)


class VisualizationTests(unittest.TestCase):
    def test_bar_counts_handle_a_column_named_count(self):
        figure = bar_chart(pd.DataFrame({"count": ["a", "a", "b"]}), "count")
        self.assertEqual(list(figure.data[0].x), ["a", "b"])
        self.assertEqual(list(figure.data[0].y), [2, 1])

    def test_bar_group_and_value_can_be_the_same_column(self):
        figure = bar_chart(pd.DataFrame({"x": [1, 1, 2]}), "x", "x", "sum")
        self.assertEqual(list(figure.data[0].x), [1, 2])
        self.assertEqual(list(figure.data[0].y), [2, 2])

    def test_bar_count_with_value_counts_observed_values(self):
        frame = pd.DataFrame({"group": ["a", "a", "b"], "value": [1.0, None, 3.0]})
        figure = bar_chart(frame, "group", "value", "count")
        self.assertEqual(list(figure.data[0].y), [1, 1])

    def test_histogram_accepts_integer_header_labels(self):
        figure = histogram(pd.DataFrame({0: [1, 2, 3]}), 0)
        self.assertEqual(list(figure.data[0].x), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
