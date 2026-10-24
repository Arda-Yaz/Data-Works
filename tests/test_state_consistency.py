import copy
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from backend.cleaning import find_mismatches_by_allowed_values, handle_missing
from backend.state import MAX_HISTORY, StateManager


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class StateConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.patcher = patch("backend.state.st.session_state", self.state)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        StateManager.init_session()

    def load(self):
        frame = pd.DataFrame({
            "value": pd.Series([1, 2, 3], dtype="Int64"),
            "label": ["a", "b", "a"],
        })
        metadata = {
            "value": {"original_dtype": "Int64", "inferred_dtype": "int64", "role": "feature", "allowed_values": ["1", "2", "3"]},
            "label": {"original_dtype": "object", "inferred_dtype": "object", "role": "feature", "allowed_values": ["a", "b"]},
        }
        StateManager.load_dataset(frame, metadata)
        StateManager.init_session()
        return frame, metadata

    def snapshot(self):
        return copy.deepcopy(dict(self.state))

    def assert_state_unchanged(self, previous):
        self.assertEqual(set(self.state), set(previous))
        for key, expected in previous.items():
            actual = self.state[key]
            if isinstance(expected, pd.DataFrame):
                assert_frame_equal(actual, expected)
            elif key == "history":
                self.assertEqual(len(actual), len(expected))
                for entry, old_entry in zip(actual, expected):
                    assert_frame_equal(entry["df"], old_entry["df"])
                    self.assertEqual({k: v for k, v in entry.items() if k != "df"},
                                     {k: v for k, v in old_entry.items() if k != "df"})
            else:
                self.assertEqual(actual, expected, key)

    def test_load_copies_dataframe_and_nested_metadata(self):
        frame, metadata = self.load()
        frame.loc[0, "value"] = 99
        metadata["value"]["allowed_values"].append("99")
        self.assertEqual(self.state.df.loc[0, "value"], 1)
        self.assertEqual(self.state.original_df.loc[0, "value"], 1)
        self.assertNotIn("99", self.state.column_meta["value"]["allowed_values"])

    def test_dropping_column_removes_metadata_and_invalid_target(self):
        self.load()
        self.state.target_column = "label"
        self.assertTrue(StateManager.apply_operation("Drop label", self.state.df.drop(columns="label")))
        self.assertEqual(list(self.state.column_meta), ["value"])
        self.assertIsNone(self.state.target_column)
        self.assertEqual(StateManager.get_columns_by_role("feature"), ["value"])

    def test_new_columns_receive_metadata(self):
        self.load()
        candidate = self.state.df.assign(added=[10.0, 20.0, 30.0])
        StateManager.apply_operation("Add feature", candidate)
        self.assertEqual(set(self.state.column_meta), set(candidate.columns))
        self.assertEqual(self.state.column_meta["added"]["cardinality"], 3)

    def test_mutation_refreshes_cardinality_and_preserves_user_rules(self):
        self.load()
        candidate = self.state.df.iloc[:1].copy()
        before = copy.deepcopy(self.state.column_meta)
        StateManager.apply_operation("Keep one row", candidate)
        self.assertEqual(self.state.column_meta["value"]["cardinality"], 1)
        self.assertEqual(self.state.column_meta["label"]["cardinality"], 1)
        self.assertEqual(self.state.column_meta["value"]["allowed_values"], before["value"]["allowed_values"])
        self.assertEqual(self.state.column_meta["value"]["original_dtype"], before["value"]["original_dtype"])
        self.assertEqual(self.state.column_meta["value"]["role"], before["value"]["role"])

    def test_apply_operation_copies_candidate_dataframe_and_metadata(self):
        self.load()
        candidate = self.state.df.copy()
        candidate.loc[0, "value"] = 10
        metadata = copy.deepcopy(self.state.column_meta)
        metadata["label"]["allowed_values"] = ["a", "b", "c"]
        StateManager.apply_operation("Edit data and schema", candidate, column_meta=metadata)
        candidate.loc[0, "value"] = 99
        metadata["label"]["allowed_values"].append("d")
        self.assertEqual(self.state.df.loc[0, "value"], 10)
        self.assertEqual(self.state.column_meta["label"]["allowed_values"], ["a", "b", "c"])

    def test_cleaning_dtype_change_clears_obsolete_conversion_request(self):
        self.load()
        self.state.column_meta["value"]["user_dtype"] = "int64"
        candidate = self.state.df.astype({"value": "Float64"})
        candidate.loc[0, "value"] = 1.5
        StateManager.apply_operation("Fractional fill", candidate)
        metadata = self.state.column_meta["value"]
        self.assertEqual(metadata["current_dtype"], "Float64")
        self.assertEqual(metadata["original_dtype"], "Int64")
        self.assertEqual(metadata["inferred_dtype"], "float64")
        self.assertIsNone(metadata["user_dtype"])

    def test_explicit_schema_keeps_successfully_applied_user_type(self):
        self.load()
        metadata = copy.deepcopy(self.state.column_meta)
        metadata["value"]["user_dtype"] = "float64"
        candidate = self.state.df.astype({"value": "Float64"})
        StateManager.apply_operation("Schema", candidate, column_meta=metadata, schema_confirmed=True)
        self.assertEqual(self.state.column_meta["value"]["current_dtype"], "Float64")
        self.assertEqual(self.state.column_meta["value"]["user_dtype"], "float64")

    def test_schema_conversion_is_undoable_with_confirmation_status(self):
        self.load()
        before = self.state.df.copy()
        metadata = copy.deepcopy(self.state.column_meta)
        metadata["value"]["user_dtype"] = "float64"
        candidate = before.astype({"value": "Float64"})
        self.assertTrue(StateManager.apply_operation(
            "Confirm schema", candidate, column_meta=metadata, schema_confirmed=True,
        ))
        self.assertTrue(self.state.schema_confirmed)
        self.assertEqual(len(self.state.history), 1)
        self.assertTrue(StateManager.undo())
        assert_frame_equal(self.state.df, before)
        self.assertFalse(self.state.schema_confirmed)
        self.assertIsNone(self.state.column_meta["value"].get("user_dtype"))

    def test_undo_restores_target_and_column_metadata(self):
        self.load()
        self.state.target_column = "label"
        self.state.schema_confirmed = True
        before = copy.deepcopy(self.state.column_meta)
        StateManager.apply_operation("Drop target", self.state.df.drop(columns="label"))
        self.assertTrue(StateManager.undo())
        self.assertEqual(self.state.target_column, "label")
        self.assertTrue(self.state.schema_confirmed)
        self.assertEqual(self.state.column_meta, before)

    def test_legacy_snapshot_also_restores_schema_and_target(self):
        self.load()
        self.state.schema_confirmed = True
        self.state.target_column = "label"
        StateManager.save_snapshot("Checkpoint")
        self.state.schema_confirmed = False
        self.state.target_column = None
        self.assertTrue(StateManager.undo())
        self.assertTrue(self.state.schema_confirmed)
        self.assertEqual(self.state.target_column, "label")

    def test_noop_does_not_add_history_or_invalidate_charts(self):
        self.load()
        self.state.viz_charts = ["saved"]
        before = self.snapshot()
        self.assertFalse(StateManager.apply_operation("No changes", self.state.df.copy()))
        self.assert_state_unchanged(before)

    def test_metadata_only_change_is_a_real_undoable_operation(self):
        self.load()
        metadata = copy.deepcopy(self.state.column_meta)
        metadata["label"]["allowed_values"] = ["a"]
        self.assertTrue(StateManager.apply_operation("Set rules", self.state.df, column_meta=metadata))
        self.assertEqual(self.state.column_meta["label"]["allowed_values"], ["a"])
        self.assertTrue(StateManager.undo())
        self.assertEqual(self.state.column_meta["label"]["allowed_values"], ["a", "b"])

    def test_invalid_dataframe_leaves_state_unchanged(self):
        self.load()
        self.state.viz_charts = ["saved"]
        before = self.snapshot()
        with self.assertRaises((TypeError, ValueError)):
            StateManager.apply_operation("Invalid data", None)
        self.assert_state_unchanged(before)

    def test_ambiguous_duplicate_column_names_are_rejected_atomically(self):
        self.load()
        before = self.snapshot()
        invalid = pd.DataFrame([[1, 2]], columns=["value", "value"])
        with self.assertRaises(ValueError):
            StateManager.apply_operation("Ambiguous columns", invalid)
        self.assert_state_unchanged(before)

    def test_malformed_metadata_leaves_state_unchanged(self):
        self.load()
        before = self.snapshot()
        candidate = self.state.df.copy()
        candidate.loc[0, "value"] = 10
        with self.assertRaises((TypeError, ValueError)):
            StateManager.apply_operation("Invalid schema", candidate, column_meta={"value": "invalid"})
        self.assert_state_unchanged(before)

    def test_history_is_bounded_and_original_dataset_is_unchanged(self):
        original, _ = self.load()
        for number in range(MAX_HISTORY + 3):
            candidate = self.state.df.copy()
            candidate.loc[0, "value"] = number + 10
            StateManager.apply_operation(f"Edit {number}", candidate)
        self.assertEqual(len(self.state.history), MAX_HISTORY)
        self.assertEqual(self.state.history[0]["name"], "Edit 3")
        assert_frame_equal(self.state.original_df, original)

    def test_invalid_nested_schema_fields_are_rejected_atomically(self):
        self.load()
        before = self.snapshot()
        for field, value in (
            ("role", "unsupported"), ("inferred_dtype", None), ("user_dtype", 42),
            ("allowed_values", "a,b"), ("allowed_values", [1, 2]), ("date_format", 42),
        ):
            with self.subTest(field=field, value=value):
                metadata = copy.deepcopy(self.state.column_meta)
                metadata["value"][field] = value
                with self.assertRaisesRegex(ValueError, "value"):
                    StateManager.apply_operation("Invalid schema", self.state.df, column_meta=metadata)
                self.assert_state_unchanged(before)

    def test_mismatch_checks_ignore_removed_columns_defensively(self):
        from backend.cleaning import find_mismatches_by_allowed_values

        frame, _ = self.load()
        result = find_mismatches_by_allowed_values(frame, {"removed": ["x"], "label": ["a"]})
        self.assertEqual(result, {"label": [1]})

    def test_numeric_rules_remain_valid_after_fractional_dtype_promotion(self):
        frame = pd.DataFrame({"value": pd.Series([1, 2, None], dtype="Int64")})
        StateManager.load_dataset(frame, {"value": {"inferred_dtype": "int64", "allowed_values": ["1", "2"]}})
        candidate = handle_missing(self.state.df, "value", "median")
        StateManager.apply_operation("Fill median", candidate)
        rules = self.state.column_meta["value"]["allowed_values"]
        self.assertEqual(rules, ["1", "2"])
        self.assertEqual(find_mismatches_by_allowed_values(self.state.df, {"value": rules}), {"value": [2]})

    def test_large_integer_rules_compare_without_float_rounding(self):
        base = 2**53 + 1
        frame = pd.DataFrame({"value": pd.Series([base, base + 1, None], dtype="Int64")})
        StateManager.load_dataset(frame, {"value": {"inferred_dtype": "int64", "allowed_values": [str(base)]}})
        result = find_mismatches_by_allowed_values(self.state.df, {"value": [str(base)]})
        self.assertEqual(result, {"value": [1]})

    def test_text_rules_preserve_leading_zero_and_literal_na_values(self):
        frame = pd.DataFrame({"code": pd.Series(["001", "1", "NA"], dtype="string")})
        StateManager.load_dataset(frame, {"code": {"inferred_dtype": "object", "allowed_values": ["001", "NA"]}})
        self.assertEqual(
            find_mismatches_by_allowed_values(self.state.df, {"code": ["001", "NA"]}),
            {"code": [1]},
        )

    def test_boolean_and_iso_date_rules_follow_column_types(self):
        frame = pd.DataFrame({
            "flag": pd.Series([True, False, None], dtype="boolean"),
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
        })
        metadata = {
            "flag": {"inferred_dtype": "bool", "allowed_values": ["true"]},
            "date": {"inferred_dtype": "datetime64[ns]", "allowed_values": ["2024-01-01"]},
        }
        StateManager.load_dataset(frame, metadata)
        self.assertEqual(
            find_mismatches_by_allowed_values(self.state.df, {"flag": ["true"], "date": ["2024-01-01"]}),
            {"flag": [1], "date": [1]},
        )

    def test_invalid_numeric_rules_are_rejected_before_any_state_change(self):
        self.load()
        self.state.viz_charts = ["saved chart"]
        before = self.snapshot()
        for rule in ("5 out of 10", "not numeric", "NaN", "inf"):
            with self.subTest(rule=rule):
                metadata = copy.deepcopy(self.state.column_meta)
                metadata["value"]["allowed_values"] = [rule]
                candidate = self.state.df.copy()
                candidate.loc[0, "value"] = 10
                with self.assertRaisesRegex(ValueError, "value"):
                    StateManager.apply_operation("Bad rule", candidate, column_meta=metadata)
                self.assert_state_unchanged(before)

    def test_mutation_and_undo_invalidate_saved_charts_and_editor_state(self):
        self.load()
        for operation in ("edit", "undo"):
            with self.subTest(operation=operation):
                self.state.viz_charts = ["stale chart"]
                self.state.schema_editor = {"edited_rows": {0: {"role": "ignore"}}}
                self.state.av_label = "stale rule"
                if operation == "edit":
                    candidate = self.state.df.copy()
                    candidate.loc[0, "value"] = 10
                    StateManager.apply_operation("Edit", candidate)
                else:
                    StateManager.undo()
                StateManager.init_session()
                self.assertFalse(self.state.get("viz_charts"))
                self.assertNotIn("schema_editor", self.state)
                self.assertNotIn("av_label", self.state)

    def test_load_replacement_resets_dataset_outputs_and_widgets(self):
        self.load()
        self.state.update({
            "viz_charts": ["old chart"], "schema_editor": {"old": True},
            "av_label": "old rule", "ml_target": "label", "ml_features": ["value"],
            "ms_value": "constant", "dup_subset": ["label"], "vz_hist_col": "value",
            "navigation": "Visualization", "theme": "dark",
        })
        StateManager.load_dataset(pd.DataFrame({"different": [4, 5]}), {})
        StateManager.init_session()
        self.assertFalse(self.state.get("viz_charts"))
        self.assertFalse(self.state.history)
        self.assertFalse(self.state.schema_confirmed)
        for key in ("schema_editor", "av_label", "ml_target", "ml_features", "ms_value", "dup_subset", "vz_hist_col"):
            self.assertNotIn(key, self.state)
        self.assertEqual(self.state.navigation, "Visualization")
        self.assertEqual(self.state.theme, "dark")
        self.assertEqual(set(self.state.column_meta), {"different"})

    def test_clear_removes_data_and_outputs_without_erasing_preferences(self):
        self.load()
        self.state.update({"viz_charts": ["old chart"], "schema_editor": {"old": True},
                           "av_label": "old rule", "navigation": "Upload", "theme": "dark"})
        StateManager.clear()
        StateManager.init_session()
        self.assertFalse(StateManager.has_dataset())
        self.assertIsNone(self.state.get("original_df"))
        self.assertFalse(self.state.get("column_meta"))
        self.assertFalse(self.state.get("history"))
        self.assertFalse(self.state.get("viz_charts"))
        self.assertNotIn("schema_editor", self.state)
        self.assertNotIn("av_label", self.state)
        self.assertEqual(self.state.navigation, "Upload")
        self.assertEqual(self.state.theme, "dark")


if __name__ == "__main__":
    unittest.main()
