"""Independent workload arithmetic and F14 gate math, without backend execution."""

import copy
import csv
import io
import unittest

from scripts.benchmark_fulfillment import assess, expected_report, generate_workload, nearest_rank


class FulfillmentBenchmarkTests(unittest.TestCase):
    def test_full_workload_has_independently_declared_nontrivial_totals(self):
        self.assertEqual(expected_report(100_000), {
            "source_rows": 100_000, "accepted_rows": 50_000, "rejected_rows": 50_000,
            "rejection_reason_assignments": 70_000, "accepted_distinct_orders": 40_000,
            "accepted_quantity": 120_000,
            "accepted_line_counts_by_status": {"fulfilled": 20_000, "pending": 20_000, "cancelled": 10_000},
            "accepted_quantity_by_status": {"fulfilled": 60_000, "pending": 30_000, "cancelled": 30_000},
        })

    def test_generator_is_deterministic_and_keeps_duplicate_and_multireason_lexemes(self):
        data = generate_workload(20)
        self.assertEqual(data, generate_workload(20))
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["order_id"], "0000000000")
        self.assertEqual(rows[0]["order_id"], rows[1]["order_id"])
        self.assertEqual(rows[0]["line_id"], rows[1]["line_id"])
        self.assertNotEqual(rows[0]["order_date"], rows[1]["order_date"])
        self.assertEqual([rows[6][field] for field in ("order_id", "sku", "quantity")], ["", "   ", "-1"])
        self.assertNotEqual(rows[0]["order_id"], rows[10]["order_id"])
        for invalid in (0, -10, 11, True):
            with self.subTest(rows=invalid), self.assertRaises(ValueError):
                generate_workload(invalid)

    def test_nearest_rank_uses_the_tenth_sample_for_p95_of_ten(self):
        self.assertEqual(nearest_rank(list(range(1, 11))), 10)
        self.assertEqual(nearest_rank([30]), 30)
        self.assertEqual(nearest_rank([3, 1, 2], percentile=0.5), 2)
        for values in ([], [float("nan")], [float("inf")], [-1], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                nearest_rank(values)

    def report(self):
        sample = {"kind": "measured", "status": "passed",
                  "run": {"execution_seconds": 30, "export_seconds": 30, "peak_rss_bytes": 1024**3},
                  "reopen": {"reopen_seconds": 30, "peak_rss_bytes": 1024**3}}
        samples = [copy.deepcopy(sample) for _ in range(11)]
        samples[0]["kind"] = "warmup"
        return {"configuration": {"rows": 100_000, "runs": 10, "warmups": 1, "include_history": False},
                "workload": {"input_bytes": 20 * 1024**2}, "samples": samples}

    def test_full_gate_passes_at_exact_limits_but_not_for_a_partial_pilot(self):
        report = self.report()
        self.assertTrue(assess(report)["f14_gate_passed"])
        report["configuration"]["rows"] = 100
        self.assertEqual(assess(report)["status"], "pilot_complete")
        self.assertFalse(assess(report)["f14_gate_passed"])
        report = self.report()
        report["samples"].pop()
        self.assertEqual(assess(report)["status"], "incomplete")
        self.assertFalse(assess(report)["f14_gate_passed"])

    def test_one_slow_or_oversized_or_failed_sample_cannot_be_hidden(self):
        mutations = [
            lambda report: report["samples"][-1]["run"].update(execution_seconds=30.001),
            lambda report: report["samples"][-1]["run"].update(export_seconds=30.001),
            lambda report: report["samples"][-1]["reopen"].update(reopen_seconds=30.001),
            lambda report: report["samples"][0]["run"].update(peak_rss_bytes=1024**3 + 1),
            lambda report: report["samples"][0].update(status="failed"),
            lambda report: report["workload"].update(input_bytes=20 * 1024**2 + 1),
        ]
        for index, mutate in enumerate(mutations):
            report = self.report()
            mutate(report)
            with self.subTest(mutation=index):
                self.assertEqual(assess(report)["status"], "failed")
                self.assertFalse(assess(report)["f14_gate_passed"])


if __name__ == "__main__":
    unittest.main()
