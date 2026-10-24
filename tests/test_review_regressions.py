import unittest
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from backend.ingestion import apply_inferred_dtypes, infer_column_meta, infer_dtype, parse_file
from backend.ml import get_available_models, train_model
from backend.preprocessing import auto_preprocess


ROOT = Path(__file__).resolve().parents[1]


def upload(text, name="example.csv"):
    stream = StringIO(text)
    stream.name = name
    return stream


def schema(dtype, **kwargs):
    return {"x": {"inferred_dtype": dtype, **kwargs}}


class IngestionTests(unittest.TestCase):
    def test_csv_preserves_identifiers_and_literal_na(self):
        df = parse_file(upload("customer_id,region\n0001,NA\n0002,EU\n"))
        self.assertEqual(df["customer_id"].tolist(), ["0001", "0002"])
        self.assertEqual(df["region"].tolist(), ["NA", "EU"])
        after = apply_inferred_dtypes(df, infer_column_meta(df))
        assert_frame_equal(df, after)

    def test_tsv_preserves_text_and_blank_cells(self):
        df = parse_file(upload("code\tregion\n001\tNULL\n002\t\n", "example.tsv"))
        self.assertEqual(df["code"].tolist(), ["001", "002"])
        self.assertEqual(df.loc[0, "region"], "NULL")
        self.assertTrue(pd.isna(df.loc[1, "region"]))

    def test_excel_preserves_text_cells(self):
        stream = BytesIO()
        pd.DataFrame({"id": ["0001", "0002"], "region": ["NA", "EU"]}).to_excel(stream, index=False)
        stream.name = "example.xlsx"
        stream.seek(0)
        df = parse_file(stream)
        self.assertEqual(df["id"].tolist(), ["0001", "0002"])
        self.assertEqual(df["region"].tolist(), ["NA", "EU"])

    def test_json_preserves_quoted_numbers_and_dates(self):
        df = parse_file(upload('[{"id":"0001","date":"2024-01-02","region":"NA"}]', "example.json"))
        self.assertEqual(df.loc[0, "id"], "0001")
        self.assertEqual(df.loc[0, "date"], "2024-01-02")

    def test_compound_values_and_ambiguous_dates_remain_text(self):
        for values in (["5 out of 10"], ["A001"], ["01/02/2024"], ["33.8 Lac", "1.2 Cr"], ["001", "002"]):
            with self.subTest(values=values):
                self.assertEqual(infer_dtype(pd.Series(values, dtype=object))[0], "object")

    def test_scientific_notation_retains_numeric_meaning(self):
        df = pd.DataFrame({"x": ["1e3", "-2e-1", None]})
        result = apply_inferred_dtypes(df, infer_column_meta(df))
        self.assertEqual(result.loc[0, "x"], 1000)
        self.assertEqual(result.loc[1, "x"], -0.2)
        self.assertTrue(pd.isna(result.loc[2, "x"]))

    def test_partial_numeric_column_is_not_automatically_coerced(self):
        values = pd.Series(["1"] * 9 + ["unknown"], dtype=object)
        self.assertEqual(infer_dtype(values)[0], "object")

    def test_string_dtype_is_supported(self):
        self.assertEqual(infer_dtype(pd.Series(["1", "2", None], dtype="string"))[0], "int64")

    def test_integer_conversion_preserves_large_integers_and_missing_values(self):
        df = pd.DataFrame({"x": ["9007199254740993", None]})
        result = apply_inferred_dtypes(df, schema("int64"))
        self.assertEqual(result.loc[0, "x"], 9007199254740993)
        self.assertTrue(pd.isna(result.loc[1, "x"]))

    def test_invalid_conversion_is_atomic_and_reports_column(self):
        df = pd.DataFrame({"valid": ["1", "2"], "x": ["1.5", "2"]})
        before = df.copy(deep=True)
        meta = {"valid": {"inferred_dtype": "int64"}, **schema("int64")}
        with self.assertRaisesRegex(ValueError, "x"):
            apply_inferred_dtypes(df, meta)
        assert_frame_equal(df, before)

    def test_invalid_boolean_does_not_become_missing(self):
        with self.assertRaisesRegex(ValueError, "x"):
            apply_inferred_dtypes(pd.DataFrame({"x": ["true", "maybe"]}), schema("bool"))

    def test_nonfinite_or_inexact_large_numeric_conversion_is_rejected(self):
        for values in (["1e999"], ["1e-999"], ["9007199254740993", "1.5"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                apply_inferred_dtypes(pd.DataFrame({"x": values}), schema("float64"))

    def test_dates_need_an_unambiguous_format(self):
        iso = pd.DataFrame({"x": ["2024-01-02", None]})
        result = apply_inferred_dtypes(iso, infer_column_meta(iso))
        self.assertEqual(result.loc[0, "x"], pd.Timestamp("2024-01-02"))
        ambiguous = pd.DataFrame({"x": ["01/02/2024"]})
        with self.assertRaises(ValueError):
            apply_inferred_dtypes(ambiguous, schema("datetime64[ns]"))
        result = apply_inferred_dtypes(ambiguous, schema("datetime64[ns]", date_format="%d/%m/%Y"))
        self.assertEqual(result.loc[0, "x"], pd.Timestamp("2024-02-01"))

    def test_id_named_column_is_not_reinterpreted(self):
        df = pd.DataFrame({"customer_id": ["1", "2", "1"]})
        after = apply_inferred_dtypes(df, infer_column_meta(df))
        assert_frame_equal(df, after)

    def test_bundled_data_preserves_text_and_missingness(self):
        with (ROOT / "surat_uncleaned.csv").open("rb") as stream:
            df = parse_file(stream)
        after = apply_inferred_dtypes(df, infer_column_meta(df))
        for col in ("property_name", "floor", "price", "square_feet"):
            pd.testing.assert_series_equal(df[col], after[col])
        pd.testing.assert_series_equal(df.isna().sum(), after.isna().sum())


class PreprocessingTests(unittest.TestCase):
    def test_same_category_has_same_representation_in_both_splits(self):
        df = pd.DataFrame({"category": ["a", "b", "c"] * 20, "target": np.arange(60)})
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
        # Make the held-out split lack the training baseline category.
        df.loc[test_idx, "category"] = ["b", "c"] * 6
        train, test, _, _, _ = auto_preprocess(df, "target", {})
        train_b = train.loc[df.loc[train.index, "category"].eq("b")].iloc[0]
        test_b = test.loc[df.loc[test.index, "category"].eq("b")].iloc[0]
        np.testing.assert_allclose(train_b, test_b)

    def test_features_are_imputed_using_training_values_only(self):
        df = pd.DataFrame({"number": pd.Series(np.arange(60), dtype="Int64"), "target": np.arange(60)})
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
        df.loc[train_idx[0], "number"] = pd.NA
        df.loc[test_idx, "number"] = 1000000
        df.loc[test_idx[0], "number"] = pd.NA
        train, test, y_train, y_test, info = auto_preprocess(df, "target", {})
        self.assertEqual(len(train) + len(test), len(df))
        self.assertEqual(len(y_train) + len(y_test), len(df))
        self.assertTrue(np.isfinite(train.to_numpy()).all())
        self.assertTrue(np.isfinite(test.to_numpy()).all())
        # A future missing value must use exactly the training-fitted transform.
        transformed = info["preprocessor"].transform(df.loc[[test_idx[0]]].drop(columns="target"))
        np.testing.assert_allclose(transformed, test.loc[[test_idx[0]]])
        expected = df.loc[train_idx, "number"].median()
        np.testing.assert_allclose(info["fill_values"]["number"], expected)

    def test_nullable_categorical_and_entirely_missing_numeric_features(self):
        df = pd.DataFrame({
            "text": pd.Series(["a", None, "b"] * 20, dtype="string"),
            "flag": pd.Series([True, None, False] * 20, dtype="boolean"),
            "empty": pd.Series([None] * 60, dtype="Int64"),
            "target": np.arange(60),
        })
        train, test, y_train, y_test, _ = auto_preprocess(df, "target", {})
        self.assertTrue(np.isfinite(train.to_numpy()).all())
        self.assertTrue(np.isfinite(test.to_numpy()).all())
        LinearRegression().fit(train, y_train).predict(test)

    def test_only_missing_targets_are_dropped(self):
        df = pd.DataFrame({"x": [1.0, None] * 30, "target": [0.0, 1.0] * 30})
        df.loc[0, "target"] = np.nan
        train, test, _, _, info = auto_preprocess(df, "target", {})
        self.assertEqual(len(train) + len(test), 59)
        self.assertEqual(info["dropped_missing_targets"], 1)

    def test_unseen_categories_do_not_refit_the_encoder(self):
        df = pd.DataFrame({"category": ["a", "b"] * 30, "target": np.arange(60)})
        _, _, _, _, info = auto_preprocess(df, "target", {})
        encoder = info["preprocessor"].named_transformers_["categorical"].named_steps["encode"]
        before = [c.copy() for c in encoder.categories_]
        result = info["preprocessor"].transform(pd.DataFrame({"category": ["new"]}))
        self.assertTrue(np.isfinite(result).all())
        for old, current in zip(before, encoder.categories_):
            np.testing.assert_array_equal(old, current)

    def test_high_cardinality_is_bounded(self):
        df = pd.DataFrame({"category": [f"value_{i}" for i in range(300)], "target": np.arange(300)})
        train, _, _, _, _ = auto_preprocess(df, "target", {})
        self.assertLessEqual(train.shape[1], 64)

    def test_empty_training_category_handles_future_values(self):
        df = pd.DataFrame({"category": pd.Series([None] * 60, dtype="string"), "target": np.arange(60)})
        train, test, _, _, info = auto_preprocess(df, "target", {})
        self.assertTrue(np.isfinite(train.to_numpy()).all())
        self.assertTrue(np.isfinite(test.to_numpy()).all())
        transformed = info["preprocessor"].transform(pd.DataFrame({"category": ["new"]}))
        self.assertTrue(np.isfinite(transformed).all())


class ModelIsolationTests(unittest.TestCase):
    def test_each_registry_request_returns_fresh_estimators(self):
        for task in ("classification", "regression"):
            first = get_available_models(task)
            second = get_available_models(task)
            for name in first:
                self.assertIsNot(first[name], second[name])
                self.assertEqual(first[name].get_params(), second[name].get_params())

    def test_training_one_run_cannot_change_another_run(self):
        first = get_available_models("regression")["Linear Regression"]
        second = get_available_models("regression")["Linear Regression"]
        x = pd.DataFrame({"x": np.arange(10, dtype=float)})
        train_model(first, x, np.arange(10, dtype=float))
        predictions = first.predict(x).copy()
        train_model(second, x, -np.arange(10, dtype=float))
        np.testing.assert_allclose(first.predict(x), predictions)


if __name__ == "__main__":
    unittest.main()
