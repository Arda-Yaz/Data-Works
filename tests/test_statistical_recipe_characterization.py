"""Characterize existing generic replay; fulfillment never applies these steps.

These frozen arithmetic examples prove today's refitting behavior. They do not
claim that generic recipes persist a fitting policy or support frozen values.
"""

import json
from pathlib import Path
import unittest

import pandas as pd

from backend.operations import replay_operations, validate_operation


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1" / "statistical_characterization.json"
EXAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))


class StatisticalRecipeCharacterizationTests(unittest.TestCase):
    def test_generic_median_replay_refits_on_each_input(self):
        example = EXAMPLES["median"]
        recipe = [{"kind": "missing", "parameters": {"strategies": {"value": "median"}}}]
        a = replay_operations(pd.DataFrame({"value": example["month_a"]}), {}, recipe)
        b = replay_operations(pd.DataFrame({"value": example["month_b"]}), {}, recipe)
        self.assertEqual(a["df"].iloc[2, 0], 25)
        self.assertEqual(b["df"]["value"].tolist(), example["refit_b"])
        self.assertNotEqual(b["df"]["value"].tolist(), example["frozen_a_applied_to_b"])
        self.assertTrue(a["statistical_operations"])
        self.assertTrue(b["statistical_operations"])

    def test_generic_iqr_replay_refits_bounds_on_each_input(self):
        example = EXAMPLES["iqr_cap"]
        recipe = [{"kind": "outliers", "parameters": {"actions": {"value": ["iqr", "cap"]}}}]
        a = replay_operations(pd.DataFrame({"value": example["month_a"]}), {}, recipe)
        b = replay_operations(pd.DataFrame({"value": example["month_b"]}), {}, recipe)
        self.assertEqual(a["df"]["value"].tolist(), [1, 2, 3, 4, 7])
        self.assertEqual(b["df"]["value"].tolist(), example["refit_b"])
        self.assertNotEqual(b["df"]["value"].tolist(), example["frozen_a_applied_to_b"])

    def test_unsupported_frozen_parameters_cannot_be_silently_applied(self):
        for operation in (
            {"kind": "missing", "parameters": {"strategies": {"value": "median"}, "policy": "frozen", "fitted_values": {"value": 25}}},
            {"kind": "outliers", "parameters": {"actions": {"value": ["iqr", "cap"]}, "policy": "frozen", "bounds": {"value": [-1, 7]}}},
        ):
            with self.subTest(kind=operation["kind"]), self.assertRaises(ValueError):
                validate_operation(operation)


if __name__ == "__main__":
    unittest.main()
