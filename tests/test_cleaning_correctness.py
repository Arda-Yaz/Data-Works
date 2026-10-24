import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.cleaning import get_missing_strategies, handle_missing, handle_outliers


class MissingValueTests(unittest.TestCase):
    def test_strategy_options_match_column_types(self):
        common = {"drop_rows", "drop_column", "mode", "constant"}
        numeric_only = {"mean", "median", "interpolate"}
        for dtype in ("Int64", "Float64", "int64", "float64"):
            with self.subTest(dtype=dtype):
                options = set(get_missing_strategies(pd.Series([1, 2], dtype=dtype)))
                self.assertTrue(common | numeric_only <= options)
        nonnumeric = [
            pd.Series(["a", None], dtype="string"),
            pd.Series([True, None], dtype="boolean"),
            pd.Series(["a", None], dtype="category"),
            pd.Series(pd.to_datetime(["2024-01-01", None])),
        ]
        for series in nonnumeric:
            with self.subTest(dtype=series.dtype):
                options = set(get_missing_strategies(series))
                self.assertTrue(common <= options)
                self.assertFalse(numeric_only & options)

    def test_fractional_mean_and_median_promote_nullable_integers(self):
        df = pd.DataFrame({"x": pd.Series([1, 2, None], dtype="Int64")})
        before = df.copy(deep=True)
        for strategy in ("mean", "median"):
            with self.subTest(strategy=strategy):
                result = handle_missing(df, "x", strategy)
                self.assertEqual(str(result["x"].dtype), "Float64")
                self.assertEqual(result["x"].tolist(), [1.0, 2.0, 1.5])
                assert_frame_equal(df, before)

    def test_integral_statistics_preserve_nullable_integer_dtype(self):
        df = pd.DataFrame({"x": pd.Series([1, 3, None], dtype="Int64")})
        for strategy in ("mean", "median", "mode"):
            with self.subTest(strategy=strategy):
                result = handle_missing(df, "x", strategy)
                self.assertEqual(str(result["x"].dtype), "Int64")
                self.assertEqual(result.loc[2, "x"], 1 if strategy == "mode" else 2)

    def test_interpolation_promotes_only_when_fractional_values_are_needed(self):
        for last, expected, dtype in ((3, 2, "Int64"), (4, 2.5, "Float64")):
            with self.subTest(last=last):
                df = pd.DataFrame({"x": pd.Series([1, None, last], dtype="Int64")})
                result = handle_missing(df, "x", "interpolate")
                self.assertEqual(result.loc[1, "x"], expected)
                self.assertEqual(str(result["x"].dtype), dtype)

    def test_text_numeric_constants_are_parsed(self):
        df = pd.DataFrame({"x": pd.Series([1, None], dtype="Int64")})
        for value, expected, dtype in (("0", 0, "Int64"), ("-2", -2, "Int64"), ("1.5", 1.5, "Float64")):
            with self.subTest(value=value):
                result = handle_missing(df, "x", "constant", value)
                self.assertEqual(result.loc[1, "x"], expected)
                self.assertEqual(str(result["x"].dtype), dtype)

    def test_float_constant_remains_numeric(self):
        df = pd.DataFrame({"x": pd.Series([1.0, None], dtype="Float64")})
        result = handle_missing(df, "x", "constant", "2.75")
        self.assertEqual(result.loc[1, "x"], 2.75)
        self.assertEqual(str(result["x"].dtype), "Float64")

    def test_large_integer_constant_is_exact(self):
        df = pd.DataFrame({"x": pd.Series([9007199254740993, None], dtype="Int64")})
        result = handle_missing(df, "x", "constant", "9007199254740995")
        self.assertEqual(result.loc[1, "x"], 9007199254740995)
        self.assertEqual(str(result["x"].dtype), "Int64")

    def test_unsafe_float_promotion_rejects_instead_of_rounding_existing_integers(self):
        df = pd.DataFrame({"x": pd.Series([9007199254740993, None], dtype="Int64")})
        before = df.copy(deep=True)
        with self.assertRaises(ValueError):
            handle_missing(df, "x", "constant", "1.5")
        assert_frame_equal(df, before)

    def test_narrow_float_constants_promote_instead_of_rounding(self):
        for dtype in ("Float32", "float32"):
            for value in ("1e-46", "16777217", "1.00000001"):
                with self.subTest(dtype=dtype, value=value):
                    df = pd.DataFrame({"x": pd.Series([1, None], dtype=dtype)})
                    before = df.copy(deep=True)
                    result = handle_missing(df, "x", "constant", value)
                    self.assertEqual(float(result.loc[1, "x"]), float(value))
                    self.assertEqual(str(result["x"].dtype), "Float64" if dtype == "Float32" else "float64")
                    assert_frame_equal(df, before)

    def test_fractional_interpolation_cannot_round_other_integer_replacements(self):
        df = pd.DataFrame({"x": pd.Series([2**54, None, 2**54 + 4, 0, None, 1], dtype="Int64")})
        before = df.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "x"):
            handle_missing(df, "x", "interpolate")
        assert_frame_equal(df, before)

    def test_boolean_constants_are_parsed(self):
        df = pd.DataFrame({"x": pd.Series([True, None], dtype="boolean")})
        for value, expected in (("true", True), ("false", False), ("1", True), ("0", False)):
            with self.subTest(value=value):
                result = handle_missing(df, "x", "constant", value)
                self.assertEqual(result.loc[1, "x"], expected)
                self.assertEqual(str(result["x"].dtype), "boolean")

    def test_datetime_constant_preserves_datetime_dtype(self):
        df = pd.DataFrame({"x": pd.to_datetime(["2024-01-01", None])})
        result = handle_missing(df, "x", "constant", "2024-03-15")
        self.assertEqual(result.loc[1, "x"], pd.Timestamp("2024-03-15"))
        self.assertEqual(result["x"].dtype, df["x"].dtype)

    def test_naive_datetime_constant_uses_column_timezone(self):
        df = pd.DataFrame({"x": pd.to_datetime(["2024-01-01", None]).tz_localize("Europe/Istanbul")})
        result = handle_missing(df, "x", "constant", "2024-03-15T12:30:00")
        self.assertEqual(result.loc[1, "x"], pd.Timestamp("2024-03-15T12:30:00", tz="Europe/Istanbul"))
        self.assertEqual(result["x"].dtype, df["x"].dtype)

    def test_aware_datetime_constant_is_rejected_for_naive_column(self):
        df = pd.DataFrame({"x": pd.to_datetime(["2024-01-01", None])})
        before = df.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "x"):
            handle_missing(df, "x", "constant", "2024-03-15T12:30:00+03:00")
        assert_frame_equal(df, before)

    def test_string_constant_remains_text(self):
        df = pd.DataFrame({"x": pd.Series(["existing", None], dtype="string")})
        result = handle_missing(df, "x", "constant", "0")
        self.assertEqual(result.loc[1, "x"], "0")
        self.assertEqual(str(result["x"].dtype), "string")

    def test_new_category_constant_can_be_added(self):
        df = pd.DataFrame({"x": pd.Series(["existing", None], dtype="category")})
        before = df.copy(deep=True)
        result = handle_missing(df, "x", "constant", "unknown")
        self.assertEqual(result.loc[1, "x"], "unknown")
        self.assertIsInstance(result["x"].dtype, pd.CategoricalDtype)
        self.assertIn("unknown", result["x"].cat.categories)
        assert_frame_equal(df, before)

    def test_numeric_category_constant_uses_category_type(self):
        df = pd.DataFrame({"x": pd.Categorical([1, None], categories=[1, 2])})
        result = handle_missing(df, "x", "constant", "2")
        self.assertEqual(result.loc[1, "x"], 2)
        self.assertEqual(result["x"].cat.categories.tolist(), [1, 2])

    def test_float_categories_reject_inexact_large_integer_constants(self):
        df = pd.DataFrame({"x": pd.Categorical([1.0, None], categories=[1.0, 2.0])})
        before = df.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "x"):
            handle_missing(df, "x", "constant", "9007199254740993")
        assert_frame_equal(df, before)

    def test_signed_category_accepts_uint64_constant_without_rounding(self):
        df = pd.DataFrame({"x": pd.Categorical([-1, None], categories=[-1, 1])})
        before = df.copy(deep=True)
        result = handle_missing(df, "x", "constant", "18446744073709551615")
        self.assertEqual(result.loc[1, "x"], 18446744073709551615)
        self.assertEqual(result.loc[0, "x"], -1)
        self.assertIsInstance(result["x"].dtype, pd.CategoricalDtype)
        assert_frame_equal(df, before)

    def test_invalid_typed_constants_report_column_and_leave_input_unchanged(self):
        cases = [
            (pd.Series([1, None], dtype="Int64"), "not a number"),
            (pd.Series([True, None], dtype="boolean"), "maybe"),
            (pd.Series(pd.to_datetime(["2024-01-01", None])), "not a date"),
        ]
        for series, value in cases:
            with self.subTest(dtype=series.dtype, value=value):
                df = pd.DataFrame({"x": series})
                before = df.copy(deep=True)
                with self.assertRaisesRegex(ValueError, "x"):
                    handle_missing(df, "x", "constant", value)
                assert_frame_equal(df, before)

    def test_nonfinite_numeric_constants_are_rejected(self):
        df = pd.DataFrame({"x": pd.Series([1.0, None], dtype="Float64")})
        for value in ("NaN", "inf", "-inf", "1e999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                handle_missing(df, "x", "constant", value)

    def test_incompatible_strategies_raise_actionable_errors(self):
        df = pd.DataFrame({"x": pd.Series(["a", None], dtype="string")})
        for strategy in ("mean", "median", "interpolate", "unknown"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, "x"):
                handle_missing(df, "x", strategy)

    def test_all_missing_statistics_raise_actionable_errors(self):
        df = pd.DataFrame({"x": pd.Series([None, None], dtype="Int64")})
        for strategy in ("mean", "median", "mode", "interpolate"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, "x"):
                handle_missing(df, "x", strategy)

    def test_large_integer_statistics_and_interpolation_remain_exact(self):
        base = 2**53
        for strategy in ("mean", "median"):
            with self.subTest(strategy=strategy):
                df = pd.DataFrame({"x": pd.Series([base + 1, base + 3, None], dtype="Int64")})
                result = handle_missing(df, "x", strategy)
                self.assertEqual(result.loc[2, "x"], base + 2)
                self.assertEqual(str(result["x"].dtype), "Int64")
        df = pd.DataFrame({"x": pd.Series([base + 1, None, base + 3], dtype="Int64")})
        self.assertEqual(handle_missing(df, "x", "interpolate").loc[1, "x"], base + 2)

    def test_narrow_integer_constant_widens_without_overflow(self):
        df = pd.DataFrame({"x": pd.Series([127, None], dtype="Int8")})
        result = handle_missing(df, "x", "constant", "128")
        self.assertEqual(result["x"].tolist(), [127, 128])
        self.assertTrue(pd.api.types.is_integer_dtype(result["x"]))

    def test_finite_large_scientific_notation_is_valid_for_float_columns(self):
        df = pd.DataFrame({"x": pd.Series([1.0, None], dtype="Float64")})
        self.assertEqual(handle_missing(df, "x", "constant", "1e20").loc[1, "x"], 1e20)

    def test_narrow_float_interpolation_does_not_round_new_values(self):
        next_value = np.nextafter(np.float32(1), np.float32(2))
        for dtype in ("float32", "Float32"):
            with self.subTest(dtype=dtype):
                df = pd.DataFrame({"x": pd.Series([1, np.nan, next_value], dtype=dtype)})
                result = handle_missing(df, "x", "interpolate")
                self.assertEqual(float(result.loc[1, "x"]), (1 + float(next_value)) / 2)
                self.assertEqual(str(result["x"].dtype).lower(), "float64")

    def test_ambiguous_local_dates_request_an_explicit_offset(self):
        df = pd.DataFrame({"x": pd.to_datetime(["2024-01-01", None]).tz_localize("Europe/Paris")})
        for value in ("2024-10-27T02:30:00", "2024-03-31T02:30:00"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "offset"):
                handle_missing(df, "x", "constant", value)
        result = handle_missing(df, "x", "constant", "2024-10-27T02:30:00+02:00")
        self.assertEqual(result.loc[1, "x"].isoformat(), "2024-10-27T02:30:00+02:00")

    def test_fractional_category_fill_keeps_large_integer_labels_exact(self):
        original = 2**53 + 1
        df = pd.DataFrame({"x": pd.Categorical.from_codes([0, -1], categories=pd.Index([original], dtype="int64"))})
        self.assertEqual(int(df.loc[0, "x"]), original)
        result = handle_missing(df, "x", "constant", "1.5")
        self.assertEqual(int(result.loc[0, "x"]), original)
        self.assertEqual(result.loc[1, "x"], 1.5)

    def test_all_missing_columns_still_accept_explicit_constants(self):
        df = pd.DataFrame({"x": pd.Series([None, None], dtype="Int64")})
        result = handle_missing(df, "x", "constant", "0")
        self.assertEqual(result["x"].tolist(), [0, 0])
        self.assertEqual(str(result["x"].dtype), "Int64")

    def test_drop_operations_preserve_input(self):
        df = pd.DataFrame({"x": pd.Series([1, None], dtype="Int64"), "y": ["a", "b"]})
        before = df.copy(deep=True)
        rows = handle_missing(df, "x", "drop_rows")
        columns = handle_missing(df, "x", "drop_column")
        self.assertEqual(rows.index.tolist(), [0])
        self.assertEqual(columns.columns.tolist(), ["y"])
        assert_frame_equal(df, before)


class OutlierTests(unittest.TestCase):
    @staticmethod
    def fractional_cap_frame():
        return pd.DataFrame({"x": pd.Series([0, 1, 2, 4, 100, None], dtype="Int64")})

    def test_fractional_caps_promote_nullable_integers(self):
        df = self.fractional_cap_frame()
        before = df.copy(deep=True)
        result = handle_outliers(df, "x", "iqr", "cap")
        self.assertEqual(result.loc[4, "x"], 8.5)
        self.assertEqual(str(result["x"].dtype), "Float64")
        self.assertTrue(pd.isna(result.loc[5, "x"]))
        assert_frame_equal(df, before)

    def test_integral_caps_preserve_nullable_integer_dtype(self):
        df = pd.DataFrame({"x": pd.Series([0, 1, 2, 3, 100, None], dtype="Int64")})
        result = handle_outliers(df, "x", "iqr", "cap")
        self.assertEqual(result.loc[4, "x"], 6)
        self.assertEqual(str(result["x"].dtype), "Int64")

    def test_large_integer_iqr_caps_use_exact_quartiles(self):
        base = 2**53
        df = pd.DataFrame({"x": pd.Series([base, base + 1, base + 2, base + 3, base + 100, None], dtype="Int64")})
        before = df.copy(deep=True)
        result = handle_outliers(df, "x", "iqr", "cap")
        self.assertEqual(int(result.loc[4, "x"]), base + 6)
        self.assertEqual(str(result["x"].dtype), "Int64")
        self.assertTrue(pd.isna(result.loc[5, "x"]))
        assert_frame_equal(df, before)

    def test_fractional_bounds_without_outliers_do_not_promote(self):
        df = pd.DataFrame({"x": pd.Series([0, 1, 2, 4, None], dtype="Int64")})
        result = handle_outliers(df, "x", "iqr", "cap")
        assert_frame_equal(result, df)

    def test_remove_keeps_rows_with_missing_values(self):
        df = self.fractional_cap_frame()
        before = df.copy(deep=True)
        result = handle_outliers(df, "x", "iqr", "remove")
        self.assertEqual(result.index.tolist(), [0, 1, 2, 3, 5])
        self.assertTrue(pd.isna(result.loc[5, "x"]))
        assert_frame_equal(df, before)

    def test_replace_nan_preserves_nullable_integer_dtype(self):
        df = self.fractional_cap_frame()
        result = handle_outliers(df, "x", "iqr", "replace_nan")
        self.assertTrue(pd.isna(result.loc[4, "x"]))
        self.assertTrue(pd.isna(result.loc[5, "x"]))
        self.assertEqual(str(result["x"].dtype), "Int64")
        self.assertEqual(result.index.tolist(), df.index.tolist())

    def test_replace_nan_preserves_large_native_integers(self):
        base = 2**53
        df = pd.DataFrame({"x": [base, base + 1, base + 2, base + 3, base + 100]})
        result = handle_outliers(df, "x", "iqr", "replace_nan")
        self.assertEqual(int(result.loc[1, "x"]), base + 1)
        self.assertTrue(pd.isna(result.loc[4, "x"]))
        self.assertEqual(str(result["x"].dtype), "Int64")

    def test_zscore_handles_nullable_masks_and_retains_missing_rows(self):
        df = pd.DataFrame({"x": pd.Series([0] * 12 + [100, None], dtype="Int64")})
        result = handle_outliers(df, "x", "zscore", "remove")
        self.assertEqual(result.index.tolist(), list(range(12)) + [13])
        self.assertTrue(pd.isna(result.loc[13, "x"]))

    def test_ignore_returns_unchanged_independent_frame(self):
        df = self.fractional_cap_frame()
        result = handle_outliers(df, "x", "iqr", "ignore")
        assert_frame_equal(result, df)
        self.assertIsNot(result, df)
        result.loc[0, "x"] = 99
        self.assertEqual(df.loc[0, "x"], 0)

    def test_constant_and_all_missing_columns_have_no_outliers(self):
        for values in ([1, 1, 1, None], [None, None]):
            for method in ("iqr", "zscore"):
                for action in ("remove", "cap", "replace_nan"):
                    with self.subTest(values=values, method=method, action=action):
                        df = pd.DataFrame({"x": pd.Series(values, dtype="Int64")})
                        assert_frame_equal(handle_outliers(df, "x", method, action), df)

    def test_invalid_arguments_are_rejected_atomically(self):
        df = self.fractional_cap_frame()
        before = df.copy(deep=True)
        for method, action in (("unsupported", "remove"), ("iqr", "unsupported")):
            with self.subTest(method=method, action=action), self.assertRaises(ValueError):
                handle_outliers(df, "x", method, action)
        assert_frame_equal(df, before)

    def test_nonnumeric_boolean_and_complex_outlier_columns_are_rejected(self):
        for series in (
            pd.Series(["a", "b", None], dtype="string"),
            pd.Series([True, False, None], dtype="boolean"),
            pd.Series([1 + 2j, 2 + 3j, np.nan]),
        ):
            with self.subTest(dtype=series.dtype):
                df = pd.DataFrame({"x": series})
                before = df.copy(deep=True)
                with self.assertRaisesRegex(ValueError, "x"):
                    handle_outliers(df, "x", "iqr", "cap")
                assert_frame_equal(df, before)


if __name__ == "__main__":
    unittest.main()
