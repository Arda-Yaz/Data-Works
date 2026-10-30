import copy
import unittest
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.ingestion import infer_column_meta
from backend.operations import apply_recipe_operation, replay_operations, validate_operation


def operation(kind, parameters, **audit):
    return {"kind": kind, "parameters": parameters, **audit}


class OperationValidationTests(unittest.TestCase):
    def test_normalization_owns_nested_parameters(self):
        raw = operation("missing", {"strategies": {"x": "constant"}, "fill_values": {"x": "0"}})
        descriptor = validate_operation(raw)
        self.assertTrue(descriptor["replayable"])
        raw["parameters"]["fill_values"]["x"] = "99"
        self.assertEqual(descriptor["parameters"]["fill_values"]["x"], "0")

    def test_duplicate_defaults_are_explicit(self):
        descriptor = validate_operation(operation("duplicates", {}))
        self.assertEqual(descriptor["parameters"], {"subset": None, "keep": "first"})

    def test_outlier_actions_have_a_stable_shape(self):
        descriptor = validate_operation(operation("outliers", {"actions": {"x": ["iqr", "cap"]}}))
        self.assertEqual(descriptor["parameters"]["actions"], {"x": ("iqr", "cap")})
        self.assertFalse(descriptor["parameters"]["include_protected"])

    def test_unknown_non_executable_and_bad_parameter_shapes_are_rejected(self):
        malformed = [
            None, [], {"kind": []}, operation("eval", {"code": "print('bad')"}),
            operation("missing", {"strategies": "x"}),
            operation("missing", {"strategies": {"x": "unknown"}}),
            operation("missing", {"strategies": {"x": "median"}, "fill_values": {"x": "0"}}),
            operation("missing", {"strategies": {"x": "constant"}, "fill_values": {"x": [1, 2]}}),
            operation("duplicates", {"subset": "x"}), operation("duplicates", {"keep": True}),
            operation("duplicates", {"subset": ["x", "x"]}),
            operation("outliers", {"actions": {"x": ("unknown", "cap")}}),
            operation("outliers", {"actions": {"x": ("iqr", "unknown")}}),
            operation("outliers", {"actions": {}, "include_protected": "yes"}),
            operation("rules", {"allowed_values": {"x": "a,b"}}),
            operation("rules", {"allowed_values": {"x": [1, 2]}}),
            operation("schema", {"column_meta": {"x": "int64"}}),
            operation("schema", {"column_meta": {}, "schema_confirmed": "yes"}),
            operation("schema", {"column_meta": {}, "apply_types": "yes"}),
            operation("duplicates", {"subset": None, "unexpected": "ignored"}),
        ]
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_operation(raw)

    def test_custom_records_are_honestly_non_replayable(self):
        raw = operation("custom", {"note": "External change"}, replayable=False)
        self.assertFalse(validate_operation(raw)["replayable"])
        for kind in ("custom", "legacy"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_operation(operation(kind, {}))

    def test_callables_and_arbitrary_objects_cannot_be_parameters(self):
        for value in (lambda: None, object()):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                validate_operation(operation("missing", {"strategies": {"x": "constant"}, "fill_values": {"x": value}}))

    def test_unused_audit_fields_are_not_executed_or_copied(self):
        class NeverCopy:
            def __deepcopy__(self, memo):
                raise AssertionError("Audit objects must not be executed")

        result = validate_operation(operation("duplicates", {}, ignored=NeverCopy()))
        self.assertNotIn("ignored", result)


class RecipeApplicationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "value": pd.Series([1, 2, None, 100, 2, 3], dtype="Int64"),
            "label": ["a", "b", "a", "bad", "b", "a"],
        })
        self.metadata = infer_column_meta(self.frame)

    def test_missing_operation_preserves_source_and_reports_statistics(self):
        before = self.frame.copy(deep=True)
        metadata = copy.deepcopy(self.metadata)
        result, explicit, confirmed, statistical = apply_recipe_operation(
            self.frame, self.metadata, operation("missing", {"strategies": {"value": "median"}}),
        )
        self.assertEqual(result.loc[2, "value"], 2)
        self.assertIsNone(explicit)
        self.assertIsNone(confirmed)
        self.assertTrue(statistical)
        assert_frame_equal(self.frame, before)
        self.assertEqual(self.metadata, metadata)

    def test_mismatch_removal_uses_positions_for_duplicate_labels(self):
        frame = pd.DataFrame({"label": ["allowed", "bad", "allowed"]}, index=["same", "same", "other"])
        metadata = infer_column_meta(frame)
        candidate, schema, _, learns = apply_recipe_operation(
            frame, metadata, operation("mismatches", {"allowed_values": {"label": ["allowed"]}}),
        )
        self.assertEqual(candidate.index.tolist(), ["same", "other"])
        self.assertEqual(candidate["label"].tolist(), ["allowed", "allowed"])
        self.assertEqual(schema["label"]["allowed_values"], ["allowed"])
        self.assertFalse(learns)
        self.assertIsNone(metadata["label"]["allowed_values"])

    def test_none_clears_rules_without_touching_values(self):
        self.metadata["label"]["allowed_values"] = ["a"]
        candidate, metadata, _, _ = apply_recipe_operation(
            self.frame, self.metadata, operation("rules", {"allowed_values": {"label": None}}),
        )
        assert_frame_equal(candidate, self.frame)
        self.assertIsNone(metadata["label"]["allowed_values"])
        self.assertEqual(self.metadata["label"]["allowed_values"], ["a"])

    def test_duplicate_subsets_keep_their_actual_label_types(self):
        frame = pd.DataFrame({5: [1, 1, 2], "5": ["a", "b", "c"]})
        result, _, _, _ = apply_recipe_operation(
            frame, infer_column_meta(frame), operation("duplicates", {"subset": [5], "keep": False}),
        )
        self.assertEqual(result.index.tolist(), [2])

    def test_safe_hashable_scalar_column_labels_remain_valid_references(self):
        for label in (Decimal("1.5"), ("tuple", 2), pd.Timestamp("2024-01-01")):
            with self.subTest(label=label):
                frame = pd.DataFrame({label: [1, 1, 2], "label": ["a", "b", "c"]})
                candidate, _, _, _ = apply_recipe_operation(
                    frame, infer_column_meta(frame), operation("duplicates", {"subset": [label]}),
                )
                self.assertEqual(candidate.index.tolist(), [0, 2])

    def test_all_kinds_reject_absent_referenced_columns(self):
        records = [
            operation("schema", {"column_meta": {"missing": {"inferred_dtype": "object"}}}),
            operation("missing", {"strategies": {"missing": "keep"}}),
            operation("duplicates", {"subset": ["missing"]}),
            operation("outliers", {"actions": {"missing": ("iqr", "ignore")}}),
            operation("rules", {"allowed_values": {"missing": ["a"]}}),
            operation("mismatches", {"allowed_values": {"missing": None}}),
        ]
        for raw in records:
            with self.subTest(kind=raw["kind"]), self.assertRaises(ValueError):
                apply_recipe_operation(self.frame, self.metadata, raw)

    def test_schema_operation_requires_complete_metadata(self):
        with self.assertRaisesRegex(ValueError, "every current column"):
            apply_recipe_operation(self.frame, self.metadata, operation("schema", {"column_meta": {"value": self.metadata["value"]}}))

    def test_metadata_only_schema_edit_does_not_apply_inferred_types(self):
        frame = pd.DataFrame({"x": ["1", "2"]})
        metadata = infer_column_meta(frame)
        metadata["x"]["role"] = "ignore"
        candidate, schema, confirmed, learns = apply_recipe_operation(
            frame, infer_column_meta(frame), operation("schema", {
                "column_meta": metadata, "schema_confirmed": False, "apply_types": False,
            }),
        )
        assert_frame_equal(candidate, frame)
        self.assertEqual(schema["x"]["role"], "ignore")
        self.assertFalse(confirmed)
        self.assertFalse(learns)

    def test_invalid_typed_rules_are_rejected_before_removal(self):
        before = self.frame.copy(deep=True)
        with self.assertRaisesRegex(ValueError, "value"):
            apply_recipe_operation(self.frame, self.metadata, operation("mismatches", {"allowed_values": {"value": ["not numeric"]}}))
        assert_frame_equal(self.frame, before)


class ReplayTests(unittest.TestCase):
    def test_complete_chain_reproduces_values_schema_and_provenance(self):
        original = pd.DataFrame({
            "value": ["1", "2", None, "100", "2", "3"],
            "label": ["a", "b", "a", "bad", "b", "a"],
        }, dtype=object)
        original_metadata = infer_column_meta(original)
        schema = copy.deepcopy(original_metadata)
        schema["value"]["user_dtype"] = "int64"
        schema["label"]["user_dtype"] = "object"
        records = [
            operation("schema", {"column_meta": schema}, name="Confirm schema", id="not executable"),
            operation("rules", {"allowed_values": {"label": ["a", "b", "bad"]}}),
            operation("missing", {"strategies": {"value": "median"}}, name="Handle missing values"),
            operation("duplicates", {"subset": ["value", "label"], "keep": "first"}),
            operation("outliers", {"actions": {"value": ("iqr", "cap")}, "include_protected": True}, name="Handle outliers"),
            operation("mismatches", {"allowed_values": {"label": ["a", "b"]}}),
        ]
        untouched_records = copy.deepcopy(records)
        before = original.copy(deep=True)
        result = replay_operations(original, original_metadata, records)
        expected = pd.DataFrame({
            "value": pd.Series([1.0, 2.0, 2.0, 3.0], index=[0, 1, 2, 5], dtype="Float64"),
            "label": pd.Series(["a", "b", "a", "a"], index=[0, 1, 2, 5], dtype=object),
        })
        assert_frame_equal(result["df"], expected)
        self.assertTrue(result["schema_confirmed"])
        self.assertEqual(result["statistical_operations"], ["Handle missing values", "Handle outliers"])
        self.assertEqual(result["column_meta"]["value"]["current_dtype"], "Float64")
        self.assertEqual(result["column_meta"]["value"]["original_dtype"], "object")
        self.assertIsNone(result["column_meta"]["value"]["user_dtype"])
        self.assertEqual(result["column_meta"]["label"]["allowed_values"], ["a", "b"])
        assert_frame_equal(original, before)
        self.assertEqual(records, untouched_records)

    def test_empty_recipe_returns_an_owned_unchanged_dataset(self):
        frame = pd.DataFrame({"x": [1, 2]})
        result = replay_operations(frame, infer_column_meta(frame), [])
        assert_frame_equal(result["df"], frame)
        result["df"].loc[0, "x"] = 99
        self.assertEqual(frame.loc[0, "x"], 1)
        self.assertFalse(result["schema_confirmed"])

    def test_custom_or_legacy_records_cannot_be_silently_skipped(self):
        frame = pd.DataFrame({"x": [1, 2]})
        for kind in ("custom", "legacy"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "operation 1"):
                replay_operations(frame, {}, [operation(kind, {}, replayable=False)])

    def test_invalid_later_operation_does_not_mutate_original_inputs(self):
        frame = pd.DataFrame({"x": pd.Series([1, None, 3], dtype="Int64")})
        before = frame.copy(deep=True)
        metadata = infer_column_meta(frame)
        records = [operation("missing", {"strategies": {"x": "median"}}), operation("duplicates", {"subset": ["absent"]})]
        with self.assertRaisesRegex(ValueError, "operation 2"):
            replay_operations(frame, metadata, records)
        assert_frame_equal(frame, before)

    def test_noop_statistics_do_not_create_false_provenance(self):
        frame = pd.DataFrame({"x": [1, 2, 3]})
        result = replay_operations(frame, {}, [operation("missing", {"strategies": {"x": "mean"}})])
        self.assertEqual(result["statistical_operations"], [])

    def test_recipe_history_must_be_a_list(self):
        with self.assertRaises(ValueError):
            replay_operations(pd.DataFrame({"x": [1]}), {}, {"kind": "duplicates"})

    def test_actual_state_operation_records_replay_after_metadata_synchronization(self):
        from backend.state import StateManager

        class SessionState(dict):
            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = value

        state = SessionState()
        with patch("backend.state.st.session_state", state):
            StateManager.init_session()
            original = pd.DataFrame({"value": ["1", None, "3", "100"], "label": ["a", "b", "a", "bad"]})
            StateManager.load_dataset(original, infer_column_meta(original))
            schema = copy.deepcopy(state.column_meta)
            schema["value"]["user_dtype"] = "int64"
            records = [
                operation("schema", {"column_meta": schema}),
                operation("missing", {"strategies": {"value": "median"}}),
                operation("mismatches", {"allowed_values": {"label": ["a", "b"]}}),
            ]
            for number, record in enumerate(records):
                candidate, metadata, confirmed, learns = apply_recipe_operation(state.df, state.column_meta, record)
                StateManager.apply_operation(
                    f"Operation {number}", candidate, column_meta=metadata,
                    schema_confirmed=confirmed, learns_from_data=learns, operation=record,
                )
            result = replay_operations(state.original_df, state.original_column_meta, state.operation_log)
            assert_frame_equal(result["df"], state.df)
            self.assertEqual(result["column_meta"], state.column_meta)
            self.assertEqual(result["schema_confirmed"], state.schema_confirmed)
            self.assertEqual(result["statistical_operations"], state.statistical_operations)


if __name__ == "__main__":
    unittest.main()
