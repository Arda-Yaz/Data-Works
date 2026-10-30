import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.ingestion import infer_column_meta
from backend.operations import apply_recipe_operation, replay_operations, validate_operation
from backend.text_transform import get_text_columns, map_categories, normalize_text


class TextTransformationTests(unittest.TestCase):
    def test_explicit_normalization_preserves_missing_and_mixed_nontext_values(self):
        frame = pd.DataFrame({"text": pd.Series(["  A\t B  ", None, pd.NA, np.nan, 12, True], dtype=object)})
        before = frame.copy(deep=True)
        result = normalize_text(frame, ["text"], strip=True, collapse_whitespace=True, case="lower")
        self.assertEqual(result.loc[0, "text"], "a b")
        self.assertIsNone(result.loc[1, "text"])
        self.assertIs(result.loc[2, "text"], pd.NA)
        self.assertTrue(np.isnan(result.loc[3, "text"]))
        self.assertEqual(result.loc[4, "text"], 12)
        self.assertIs(result.loc[5, "text"], True)
        assert_frame_equal(frame, before)

    def test_no_options_is_a_true_value_and_dtype_noop(self):
        frame = pd.DataFrame({"text": pd.Series([" A ", None], dtype="string")})
        result = normalize_text(frame, ["text"])
        assert_frame_equal(frame, result)
        result.loc[0, "text"] = "outside"
        self.assertEqual(frame.loc[0, "text"], " A ")

    def test_collapse_and_strip_are_independent(self):
        frame = pd.DataFrame({"text": ["  a\t  b  "]})
        self.assertEqual(normalize_text(frame, ["text"], collapse_whitespace=True).loc[0, "text"], " a b ")
        self.assertEqual(normalize_text(frame, ["text"], strip=True).loc[0, "text"], "a\t  b")

    def test_unicode_forms_and_casefold_are_explicit(self):
        frame = pd.DataFrame({"text": ["Straße", "Ａ", "e\u0301"]})
        result = normalize_text(frame, ["text"], case="casefold", unicode_form="NFKC")
        self.assertEqual(result["text"].tolist(), ["strasse", "a", "é"])
        self.assertEqual(normalize_text(frame, ["text"], unicode_form="NFC").loc[1, "text"], "Ａ")

    def test_category_collision_keeps_order_and_unused_levels(self):
        frame = pd.DataFrame({"label": pd.Categorical([" A ", "a", None], categories=[" A ", "a", " UNUSED "], ordered=True)})
        result = normalize_text(frame, ["label"], strip=True, case="lower")
        self.assertEqual(result["label"].cat.categories.tolist(), ["a", "unused"])
        self.assertTrue(result["label"].cat.ordered)
        self.assertEqual(result["label"].iloc[:2].tolist(), ["a", "a"])
        self.assertTrue(pd.isna(result["label"].iloc[2]))
        self.assertEqual(frame["label"].cat.categories.tolist(), [" A ", "a", " UNUSED "])

    def test_mapping_is_exact_simultaneous_and_retains_unmatched(self):
        # This case promises preservation of an object-stored None. pandas 3
        # otherwise infers StringDtype and converts None before our operation.
        frame = pd.DataFrame({"label": pd.Series(["a", "b", " a", "c", None], dtype=object)})
        result = map_categories(frame, "label", {"a": "b", "b": "c"})
        self.assertEqual(result["label"].tolist(), ["b", "c", " a", "c", None])
        self.assertEqual(frame["label"].tolist(), ["a", "b", " a", "c", None])

    def test_string_storage_and_missing_sentinel_survive_normalization_and_mapping(self):
        for marker in (pd.NA, np.nan):
            with self.subTest(missing_sentinel=repr(marker)):
                dtype = pd.StringDtype(storage="python", na_value=marker)
                frame = pd.DataFrame({"label": pd.Series([" A ", None, "b"], dtype=dtype)})
                frame.index = [4, 4, 9]
                before = frame.copy(deep=True)
                normalized = normalize_text(frame, ["label"], strip=True, case="lower")
                mapped = map_categories(normalized, "label", {"a": "b", "b": "a"})
                for result, expected_values in ((normalized, ["a", "b"]), (mapped, ["b", "a"])):
                    self.assertEqual(result["label"].dtype, dtype)
                    self.assertEqual(result["label"].dtype.storage, dtype.storage)
                    self.assertEqual(result.index.tolist(), [4, 4, 9])
                    self.assertEqual(result["label"].iloc[[0, 2]].tolist(), expected_values)
                    if marker is pd.NA:
                        self.assertIs(result["label"].iloc[1], pd.NA)
                    else:
                        self.assertTrue(np.isnan(result["label"].iloc[1]))
                assert_frame_equal(frame, before)

    def test_mapping_preserves_string_storage_and_duplicate_index_labels(self):
        frame = pd.DataFrame({"label": pd.Series(["a", pd.NA, "b"], dtype="string")})
        frame.index = [4, 4, 9]
        result = map_categories(frame, "label", {"a": "b", "b": "a"})
        self.assertEqual(result.index.tolist(), [4, 4, 9])
        self.assertEqual(result["label"].dtype, frame["label"].dtype)
        self.assertEqual(result["label"].iloc[[0, 2]].tolist(), ["b", "a"])
        self.assertTrue(pd.isna(result["label"].iloc[1]))

    def test_mixed_categorical_values_are_not_stringified(self):
        frame = pd.DataFrame({"label": pd.Categorical([1, "A", None], categories=[1, "A", "B"])})
        result = map_categories(frame, "label", {"A": "a"})
        self.assertEqual(result["label"].cat.categories.tolist(), [1, "a", "B"])
        self.assertEqual(result["label"].iloc[0], 1)

    def test_empty_text_storage_is_supported_without_inventing_values(self):
        frame = pd.DataFrame({"text": pd.Series([None, None], dtype=object), "numeric": pd.Series([1, 2], dtype=object)})
        self.assertEqual(get_text_columns(frame), ["text"])
        assert_frame_equal(normalize_text(frame, ["text"], strip=True), frame)

    def test_invalid_columns_and_options_fail_without_mutation(self):
        frame = pd.DataFrame({"label": ["a", "b"], "numeric": [1, 2]})
        before = frame.copy(deep=True)
        invalid = [([], {}), (["label", "label"], {}), (["missing"], {}), (["numeric"], {}),
                   (["label"], {"strip": 1}), (["label"], {"case": "title"}),
                   (["label"], {"unicode_form": "NFD"})]
        for columns, options in invalid:
            with self.subTest(columns=columns, options=options), self.assertRaises(ValueError):
                normalize_text(frame, columns, **options)
        for mapping in ([], {"a": None}, {1: "a"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                map_categories(frame, "label", mapping)
        assert_frame_equal(frame, before)


class TextRecipeTests(unittest.TestCase):
    def test_recipe_validation_owns_mapping_and_expands_normalization_defaults(self):
        raw = {"kind": "category_map", "parameters": {"column": "label", "mapping": {"a": "b"}}}
        result = validate_operation(raw)
        raw["parameters"]["mapping"]["a"] = "outside"
        self.assertEqual(result["parameters"]["mapping"], {"a": "b"})
        result = validate_operation({"kind": "text_normalize", "parameters": {"columns": ["label"]}})
        self.assertEqual(result["parameters"], {"columns": ["label"], "strip": False, "collapse_whitespace": False, "case": "keep", "unicode_form": None})

    def test_recipe_validation_rejects_bad_shapes_and_unsupported_options(self):
        invalid = [
            {"kind": "text_normalize", "parameters": {"columns": "label"}},
            {"kind": "text_normalize", "parameters": {"columns": ["label"], "regex": "x"}},
            {"kind": "text_normalize", "parameters": {"columns": ["label"], "case": "exec"}},
            {"kind": "category_map", "parameters": {"column": "label", "mapping": {"a": None}}},
        ]
        for recipe in invalid:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                validate_operation(recipe)

    def test_allowed_rules_follow_normalization_and_mapping_without_statistical_flags(self):
        frame = pd.DataFrame({"label": [" A ", "a", "B"]})
        metadata = infer_column_meta(frame)
        metadata["label"]["allowed_values"] = [" A ", "a", "B"]
        before = copy.deepcopy(metadata)
        recipe = {"kind": "text_normalize", "parameters": {"columns": ["label"], "strip": True, "case": "lower"}}
        candidate, changed, confirmed, learns = apply_recipe_operation(frame, metadata, recipe)
        self.assertEqual(candidate["label"].tolist(), ["a", "a", "b"])
        self.assertEqual(changed["label"]["allowed_values"], ["a", "b"])
        self.assertIsNone(confirmed)
        self.assertFalse(learns)
        self.assertEqual(metadata, before)
        _, mapped, _, learns = apply_recipe_operation(candidate, changed, {"kind": "category_map", "parameters": {"column": "label", "mapping": {"a": "b"}}})
        self.assertEqual(mapped["label"]["allowed_values"], ["b"])
        self.assertFalse(learns)

    def test_state_history_and_recipe_replay_match_and_undo_restores_rules(self):
        from backend.state import StateManager

        class State(dict):
            def __getattr__(self, key):
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        state = State()
        with patch("backend.state.st.session_state", state):
            StateManager.init_session()
            frame = pd.DataFrame({"label": pd.Series([" A ", "B", None], dtype="string")}, index=[0, 1, 2])
            frame.index = [5, 5, 8]
            metadata = infer_column_meta(frame)
            metadata["label"]["allowed_values"] = [" A ", "B"]
            StateManager.load_dataset(frame, metadata)
            recipe = {"kind": "text_normalize", "parameters": {"columns": ["label"], "strip": True, "case": "lower"}}
            candidate, updated, confirmed, learns = apply_recipe_operation(state.df, state.column_meta, recipe)
            StateManager.apply_operation("Normalize text", candidate, column_meta=updated, schema_confirmed=confirmed, learns_from_data=learns, operation=recipe)
            replayed = replay_operations(state.original_df, state.original_column_meta, state.operation_log)
            assert_frame_equal(replayed["df"], state.df)
            self.assertEqual(replayed["column_meta"], state.column_meta)
            self.assertEqual(state.statistical_operations, [])
            StateManager.undo()
            assert_frame_equal(state.df, frame)
            self.assertEqual(state.column_meta["label"]["allowed_values"], [" A ", "B"])


if __name__ == "__main__":
    unittest.main()
