import unittest
from unittest.mock import patch

import pandas as pd

from backend.state import StateManager


class SessionState(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class ModelingProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.patcher = patch("backend.state.st.session_state", self.state)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        StateManager.init_session()
        StateManager.load_dataset(pd.DataFrame({"x": [1.0, None, 3.0], "target": [0, 1, 0]}), {})

    def fill(self):
        return self.state.df.fillna({"x": 2.0})

    def test_statistical_fills_are_tracked_and_undo_restores_prior_provenance(self):
        StateManager.apply_operation("Median fill", self.fill(), learns_from_data=True)
        self.assertEqual(self.state.statistical_operations, ["Median fill"])
        StateManager.apply_operation("Keep first rows", self.state.df.iloc[:2])
        self.assertEqual(self.state.statistical_operations, ["Median fill"])
        self.assertTrue(StateManager.undo())
        self.assertEqual(self.state.statistical_operations, ["Median fill"])
        self.assertTrue(StateManager.undo())
        self.assertEqual(self.state.statistical_operations, [])

    def test_constant_fill_does_not_mark_statistical_learning(self):
        StateManager.apply_operation("Constant fill", self.fill())
        self.assertEqual(self.state.statistical_operations, [])

    def test_noop_does_not_create_a_false_leakage_warning(self):
        self.assertFalse(StateManager.apply_operation("No changes", self.state.df, learns_from_data=True))
        self.assertEqual(self.state.statistical_operations, [])
        self.assertEqual(self.state.history, [])

    def test_loading_or_clearing_dataset_resets_statistical_history(self):
        StateManager.apply_operation("Median fill", self.fill(), learns_from_data=True)
        StateManager.load_dataset(pd.DataFrame({"other": [1, 2]}), {})
        self.assertEqual(self.state.statistical_operations, [])
        StateManager.apply_operation("Statistical operation", self.state.df.iloc[:1], learns_from_data=True)
        StateManager.clear()
        self.assertEqual(self.state.statistical_operations, [])

    def test_widget_reset_does_not_erase_statistical_provenance(self):
        StateManager.apply_operation("Median fill", self.fill(), learns_from_data=True)
        StateManager.init_session()
        self.assertEqual(self.state.statistical_operations, ["Median fill"])

    def test_invalid_flag_is_rejected_without_any_data_changes(self):
        with self.assertRaises(ValueError):
            StateManager.apply_operation("Invalid operation", self.fill(), learns_from_data="yes")
        self.assertTrue(pd.isna(self.state.df.loc[1, "x"]))
        self.assertEqual(self.state.history, [])
        self.assertEqual(self.state.statistical_operations, [])

    def test_legacy_history_is_flagged_conservatively(self):
        self.state.pop("statistical_operations")
        self.state.history = [{"name": "Handle missing values"}]
        StateManager.init_session()
        self.assertEqual(self.state.statistical_operations, ["Handle missing values"])

    def test_legacy_undo_does_not_claim_provenance_is_known_to_be_clean(self):
        StateManager.apply_operation("Median fill", self.fill(), learns_from_data=True)
        StateManager.save_snapshot("Older checkpoint")
        self.state.history[-1].pop("statistical_operations")
        StateManager.undo()
        self.assertEqual(self.state.statistical_operations, ["Median fill"])


if __name__ == "__main__":
    unittest.main()
