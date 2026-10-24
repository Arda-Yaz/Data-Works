import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.preprocessing import auto_preprocess


class MLSplitTests(unittest.TestCase):
    def classification_frame(self):
        return pd.DataFrame({"value": np.arange(60, dtype=float), "target": ["a", "b", "c"] * 20})

    def regression_frame(self):
        return pd.DataFrame({"value": np.arange(40, dtype=float), "target": np.arange(40, dtype=float) / 3})

    def test_stratified_split_keeps_all_classes_in_both_sets(self):
        df = self.classification_frame()
        train, test, y_train, y_test, info = auto_preprocess(
            df, "target", {}, problem_type="classification", split_strategy="stratified",
        )
        self.assertEqual(set(y_train), {"a", "b", "c"})
        self.assertEqual(set(y_test), {"a", "b", "c"})
        self.assertEqual((len(train), len(test)), (48, 12))
        self.assertEqual(info["split_strategy"], "stratified")

    def test_stratified_split_is_reproducible(self):
        df = self.classification_frame()
        first = auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="stratified")
        second = auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="stratified")
        assert_frame_equal(first[0], second[0])
        assert_frame_equal(first[1], second[1])

    def test_unused_categorical_target_levels_do_not_prevent_stratification(self):
        df = self.classification_frame()
        df["target"] = pd.Categorical(df["target"], categories=["a", "b", "c", "unused"])
        _, _, y_train, y_test, _ = auto_preprocess(
            df, "target", {}, problem_type="classification", split_strategy="stratified",
        )
        self.assertEqual(set(y_train), {"a", "b", "c"})
        self.assertEqual(set(y_test), {"a", "b", "c"})

    def test_stratification_rejects_singleton_classes_without_fallback(self):
        df = pd.DataFrame({"value": range(10), "target": ["a"] * 9 + ["b"]})
        with self.assertRaisesRegex(ValueError, "two rows"):
            auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="stratified")

    def test_stratification_rejects_split_too_small_for_classes(self):
        df = pd.DataFrame({"value": range(12), "target": ["a", "b", "c"] * 4})
        with self.assertRaisesRegex(ValueError, "one row per class"):
            auto_preprocess(df, "target", {}, test_size=0.1, problem_type="classification", split_strategy="stratified")

    def test_stratification_rejects_actual_minority_class_omission(self):
        df = pd.DataFrame({"value": np.arange(100), "target": ["rare"] * 2 + ["common"] * 98})
        with self.assertRaisesRegex(ValueError, "stratified test partition is missing"):
            auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="stratified")

    def test_group_split_has_disjoint_groups_and_excludes_group_feature(self):
        df = self.regression_frame().assign(group=np.repeat(["a", "b", "c", "d"], 10))
        train, test, _, _, info = auto_preprocess(
            df, "target", {}, problem_type="regression", split_strategy="group", group_col="group",
        )
        self.assertFalse(set(df.loc[train.index, "group"]) & set(df.loc[test.index, "group"]))
        self.assertIn("group", info["dropped"])
        self.assertNotIn("group", info["encoded"])
        self.assertEqual(info["train_rows"] + info["test_rows"], len(df))

    def test_group_split_rejects_missing_or_single_group_ids(self):
        for groups in (["a"] * 40, ["a"] * 39 + [None]):
            with self.subTest(groups=groups[-2:]), self.assertRaisesRegex(ValueError, "group|Grouping"):
                auto_preprocess(
                    self.regression_frame().assign(group=groups), "target", {},
                    problem_type="regression", split_strategy="group", group_col="group",
                )

    def test_group_split_handles_mixed_scalar_group_identity_without_string_collision(self):
        df = self.regression_frame().assign(group=[1, "1", 2, "2"] * 10)
        train, test, _, _, _ = auto_preprocess(
            df, "target", {}, problem_type="regression", split_strategy="group", group_col="group",
        )
        self.assertFalse(set(df.loc[train.index, "group"]) & set(df.loc[test.index, "group"]))

    def test_chronological_split_sorts_and_keeps_equal_times_together(self):
        df = self.regression_frame().assign(time=np.repeat(np.arange(10), 4)).sample(frac=1, random_state=1)
        train, test, _, _, info = auto_preprocess(
            df, "target", {}, test_size=0.25, problem_type="regression", split_strategy="chronological", time_col="time",
        )
        self.assertLess(df.loc[train.index, "time"].max(), df.loc[test.index, "time"].min())
        self.assertNotIn("time", info["scaled"])
        self.assertEqual(len(test), 12)
        self.assertTrue(any("equal timestamps" in warning for warning in info["warnings"]))

    def test_chronological_split_accepts_iso_dates_and_uses_training_median(self):
        df = self.regression_frame().assign(time=pd.date_range("2024-01-01", periods=40).strftime("%Y-%m-%d"))
        df.loc[0, "value"] = np.nan
        df.loc[32:, "value"] = 1000000
        train, test, _, _, info = auto_preprocess(
            df, "target", {}, problem_type="regression", split_strategy="chronological", time_col="time",
        )
        self.assertEqual(train.index.tolist(), list(range(32)))
        self.assertEqual(test.index.tolist(), list(range(32, 40)))
        self.assertEqual(info["fill_values"]["value"], df.loc[1:31, "value"].median())

    def test_chronological_split_rejects_ambiguous_missing_and_constant_times(self):
        for times in (["01/02/2024"] * 40, [None] * 40, [1] * 40, [0] * 39 + [np.inf]):
            with self.subTest(times=times[-2:]), self.assertRaises(ValueError):
                auto_preprocess(
                    self.regression_frame().assign(time=times), "target", {},
                    problem_type="regression", split_strategy="chronological", time_col="time",
                )

    def test_chronological_split_does_not_split_an_oversized_first_time_group(self):
        df = self.regression_frame().assign(time=[0] * 39 + [1])
        with self.assertRaisesRegex(ValueError, "no training rows"):
            auto_preprocess(df, "target", {}, problem_type="regression", split_strategy="chronological", time_col="time")

    def test_chronological_split_rejects_strings_that_parse_as_missing_timestamps(self):
        for marker in ("NaT", "NaN", "nan", ""):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "missing timestamps"):
                df = self.regression_frame().assign(time=pd.date_range("2024-01-01", periods=40).strftime("%Y-%m-%d"))
                df.loc[0, "time"] = marker
                auto_preprocess(df, "target", {}, problem_type="regression", split_strategy="chronological", time_col="time")

    def test_chronological_classification_rejects_unseen_training_class(self):
        df = self.regression_frame().assign(time=range(40), target=["a"] * 32 + ["b"] * 8)
        with self.assertRaisesRegex(ValueError, "training partition is missing"):
            auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="chronological", time_col="time")

    def test_test_class_absence_is_reported_without_invalidating_temporal_split(self):
        df = self.regression_frame().assign(time=range(40), target=["a", "b"] * 16 + ["a"] * 8)
        _, _, y_train, y_test, info = auto_preprocess(
            df, "target", {}, problem_type="classification", split_strategy="chronological", time_col="time",
        )
        self.assertEqual(set(y_train), {"a", "b"})
        self.assertEqual(set(y_test), {"a"})
        self.assertTrue(any("does not contain every class" in warning for warning in info["warnings"]))

    def test_target_validation_rejects_one_class_fractional_and_nonfinite_values(self):
        cases = [(["a"] * 40, "classification"), (np.arange(40) / 3, "classification"),
                 ([1.0] * 39 + [np.inf], "regression"), ([None] * 40, "regression"),
                 (["1", "2"] * 20, "regression")]
        for target, problem_type in cases:
            with self.subTest(problem_type=problem_type), self.assertRaises(ValueError):
                auto_preprocess(self.regression_frame().assign(target=target), "target", {}, problem_type=problem_type)

    def test_missing_features_are_retained_and_only_missing_targets_are_removed(self):
        df = self.regression_frame()
        df.loc[::3, "value"] = np.nan
        df.loc[0, "target"] = np.nan
        train, test, _, _, info = auto_preprocess(df, "target", {}, problem_type="regression")
        self.assertEqual(len(train) + len(test), 39)
        self.assertEqual(info["dropped_missing_targets"], 1)
        self.assertTrue(np.isfinite(train.to_numpy()).all())
        self.assertTrue(np.isfinite(test.to_numpy()).all())

    def test_object_numeric_infinity_has_an_actionable_target_error(self):
        for values in ([1, np.inf] * 20, ["a", np.inf] * 20):
            with self.subTest(values=values[:2]), self.assertRaisesRegex(ValueError, "target contains non-finite"):
                df = self.regression_frame().assign(target=pd.Series(values, dtype=object))
                auto_preprocess(df, "target", {}, problem_type="classification", split_strategy="stratified")

    def test_constants_are_dropped_based_on_training_data_only(self):
        df = self.regression_frame().assign(constant=1.0, empty=pd.Series([None] * 40, dtype="Int64"))
        _, _, _, _, info = auto_preprocess(df, "target", {}, problem_type="regression")
        self.assertEqual(set(info["dropped_constants"]), {"constant", "empty"})
        self.assertNotIn("constant", info["scaled"])
        self.assertNotIn("empty", info["scaled"])

    def test_all_constant_training_features_fail_actionably(self):
        df = pd.DataFrame({"constant": [1.0] * 40, "target": range(40)})
        with self.assertRaisesRegex(ValueError, "No informative training features"):
            auto_preprocess(df, "target", {}, problem_type="regression")

    def test_nonfinite_and_complex_features_are_rejected(self):
        for values in ([1.0] * 39 + [np.inf], [1 + 2j, 2 + 3j] * 20):
            with self.subTest(values=values[-1]), self.assertRaises(ValueError):
                auto_preprocess(self.regression_frame().assign(value=values), "target", {}, problem_type="regression")

    def test_unknown_or_incompatible_split_options_do_not_fallback(self):
        for kwargs in ({"split_strategy": "unknown"}, {"split_strategy": "stratified"},
                       {"split_strategy": "group", "group_col": "missing"},
                       {"split_strategy": "chronological", "time_col": "target"}, {"test_size": 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                auto_preprocess(self.regression_frame(), "target", {}, problem_type="regression", **kwargs)

    def test_failed_preprocessing_does_not_modify_the_input(self):
        df = self.regression_frame().assign(value=np.inf)
        before = df.copy(deep=True)
        with self.assertRaises(ValueError):
            auto_preprocess(df, "target", {}, problem_type="regression")
        assert_frame_equal(df, before)

    def test_integer_feature_labels_select_actual_columns_and_keep_meaningful_names(self):
        df = pd.DataFrame({"target": np.arange(40, dtype=float), 5: np.arange(40, dtype=float), 9: ["a", "b"] * 20})
        original = df.copy(deep=True)
        train, test, _, _, info = auto_preprocess(df, "target", {}, problem_type="regression")
        self.assertIn("numeric__5", train.columns)
        self.assertTrue(any("9" in name and "a" in name for name in train.columns))
        self.assertEqual(info["preprocessor"].get_feature_names_out().tolist(), train.columns.tolist())
        reordered = df.loc[test.index, [9, 5]]
        np.testing.assert_allclose(info["preprocessor"].transform(reordered), test.to_numpy())
        assert_frame_equal(df, original)

    def test_mixed_string_and_integer_feature_labels_are_supported(self):
        df = pd.DataFrame({"target": np.arange(40, dtype=float), 5: np.arange(40, dtype=float), "kind": ["a", "b"] * 20})
        train, test, _, _, info = auto_preprocess(df, "target", {}, problem_type="regression")
        self.assertIn("numeric__5", train.columns)
        self.assertTrue(any("kind" in name for name in train.columns))
        np.testing.assert_allclose(info["preprocessor"].transform(df.loc[test.index, ["kind", 5]]), test.to_numpy())

    def test_distinct_labels_with_same_text_do_not_collide_and_preprocessor_is_reusable(self):
        import pickle
        from sklearn.base import clone

        df = pd.DataFrame({"target": np.arange(40, dtype=float), 5: np.arange(40, dtype=float), "5": ["a", "b"] * 20})
        train, test, _, _, info = auto_preprocess(df, "target", {}, problem_type="regression")
        self.assertTrue(train.columns.is_unique)
        self.assertTrue(any("int:5" in name for name in train.columns))
        self.assertTrue(any("str:" in name for name in train.columns))
        restored = pickle.loads(pickle.dumps(info["preprocessor"]))
        np.testing.assert_allclose(restored.transform(df.loc[test.index, ["5", 5]]), test.to_numpy())
        fresh = clone(info["preprocessor"])
        fresh.fit_transform(df[[5, "5"]])
        self.assertTrue(np.isfinite(fresh.transform(df[["5", 5]])).all())


if __name__ == "__main__":
    unittest.main()
