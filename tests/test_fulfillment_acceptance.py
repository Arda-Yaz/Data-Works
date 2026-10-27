"""Independent fulfillment acceptance requirements from the Phase 1 oracle.

These tests intentionally fail until the production fulfillment engine exists.
They do not regenerate, modify, skip, or bless the hand-authored expectations.
"""

import copy
import csv
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "fulfillment_v1"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
COLUMNS = EXPECTED["output_columns"]


def csv_bytes(rows, columns=COLUMNS):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def order_row(number, *, quantity="1", date="2026-10-01", status="pending", sku="sku-01"):
    return [f"{number:06d}", "01", date, sku, quantity, status]


class FulfillmentAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = importlib.import_module("backend.fulfillment")

    def run_bytes(self, data, plan=None):
        plan = self.engine.default_plan() if plan is None else plan
        source = self.engine.ingest_csv(data)
        return source, self.engine.execute(source, plan)

    def assert_completed(self, source, result, expected):
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["source_id"], source["source_id"])
        self.assertEqual(result["engine_version"], 1)
        self.assertEqual(result["report"], expected["report"])
        accepted = [{"source_record": item["source_record"], "values": item["values"]}
                    for item in result["accepted"]]
        self.assertEqual(accepted, expected["accepted"])
        rejected = []
        for item in result["rejected"]:
            actual = {"source_record": item["source_record"], "reasons": sorted(item["reasons"])}
            if "duplicate_of_source_record" in item:
                actual["duplicate_of_source_record"] = item["duplicate_of_source_record"]
            rejected.append(actual)
        wanted = [{**item, "reasons": sorted(item["reasons"])} for item in expected["rejected"]]
        self.assertEqual(rejected, wanted)
        accepted_ids = {item["row_id"] for item in result["accepted"]}
        rejected_ids = {item["row_id"] for item in result["rejected"]}
        self.assertFalse(accepted_ids & rejected_ids)
        self.assertEqual(accepted_ids | rejected_ids, set(source["row_ids"]))
        self.assertEqual(len(accepted_ids), len(result["accepted"]))
        self.assertEqual(len(rejected_ids), len(result["rejected"]))
        for item in result["accepted"] + result["rejected"]:
            self.assertEqual(item["row_id"], source["row_ids"][item["source_record"] - 1])
        for item in result["rejected"]:
            raw = dict(zip(source["columns"], source["rows"][item["source_record"] - 1]))
            self.assertEqual(item["raw"], raw)
            self.assertEqual(len(item["reasons"]), len(set(item["reasons"])))
            self.assertEqual(set(item["reasons"]), {issue["code"] for issue in item["issues"]})
            self.assertTrue(all(issue["stage"] and issue["columns"] for issue in item["issues"]))
            if "duplicate_of_source_record" in item:
                self.assertEqual(item["duplicate_of_row_id"], source["row_ids"][item["duplicate_of_source_record"] - 1])

    def assert_noncommitting(self, result, status):
        self.assertEqual(result["status"], status)
        self.assertIsNone(result["accepted"])
        self.assertIsNone(result["rejected"])
        self.assertIsNone(result["report"])

    def test_f01_f05_f06_f07_complete_fixtures_match_independent_oracle(self):
        for filename, expected in EXPECTED["completed_runs"].items():
            with self.subTest(input=filename):
                source, result = self.run_bytes((FIXTURES / filename).read_bytes())
                self.assert_completed(source, result, expected)

    def test_f03_f04_f08_f09_noncommitting_fixtures_explain_their_outcomes(self):
        for filename, expected in EXPECTED["noncommitting_attempts"].items():
            with self.subTest(input=filename):
                source, result = self.run_bytes((FIXTURES / filename).read_bytes())
                self.assert_noncommitting(result, expected["outcome"])
                self.assertEqual(result["source_id"], source["source_id"])
                issues = [issue for issue in result["issues"] if issue["code"] == expected["issue"]]
                self.assertTrue(issues, f"Missing actionable issue: {expected['issue']}")
                for field in ("columns", "key", "source_records", "differing_columns", "normalized_value"):
                    if field in expected:
                        self.assertTrue(any(issue.get(field) == expected[field] for issue in issues), field)

    def test_f02_saved_plan_processes_month_b_in_a_fresh_ui_independent_process(self):
        plan = self.engine.default_plan()
        a_source = self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        a_result = self.engine.execute(a_source, plan)
        self.assert_completed(a_source, a_result, EXPECTED["completed_runs"]["month_a.csv"])
        saved_digest = self.engine.plan_digest(plan)
        code = """
import json, sys
from pathlib import Path
from backend.fulfillment import loads_plan, plan_digest, ingest_csv, execute
plan = loads_plan(Path(sys.argv[1]).read_bytes())
before = plan_digest(plan)
source = ingest_csv(Path(sys.argv[2]).read_bytes())
result = execute(source, plan)
assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
source_evidence = {key: source[key] for key in ('source_id', 'columns', 'rows', 'row_ids')}
print(json.dumps({'source': source_evidence, 'result': result, 'before': before, 'after': plan_digest(plan)}, allow_nan=False))
"""
        with tempfile.TemporaryDirectory(prefix="fulfillment-plan-") as directory:
            plan_path = Path(directory) / "plan.json"
            plan_path.write_bytes(self.engine.dumps_plan(plan))
            completed = subprocess.run(
                [sys.executable, "-c", code, str(plan_path), str(FIXTURES / "month_b.csv")],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        child = json.loads(completed.stdout)
        self.assertEqual(child["before"], saved_digest)
        self.assertEqual(child["after"], saved_digest)
        self.assert_completed(child["source"], child["result"], EXPECTED["completed_runs"]["month_b.csv"])
        self.assertNotEqual(child["source"]["source_id"], a_source["source_id"])
        self.assertFalse(set(child["source"]["row_ids"]) & set(a_source["row_ids"]))

    def test_f11_project_archive_reopens_in_a_fresh_process_then_reuses_its_saved_plan(self):
        from backend.fulfillment_state import new_project, set_source, run_project, export_project
        from backend.project_io import dumps_project

        a_source = self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        project = run_project(set_source(new_project(), a_source))
        saved_attempt = copy.deepcopy(project["committed"]["attempt"])
        saved_digest = self.engine.plan_digest(project["draft_plan"])
        code = """
import json, sys
from pathlib import Path
from backend.fulfillment import ingest_csv, plan_digest
from backend.fulfillment_state import validate_project, set_source, run_project, current_result
from backend.project_io import loads_project
payload = loads_project(Path(sys.argv[1]).read_bytes())
opened = validate_project(payload['fulfillment'])
before = plan_digest(opened['draft_plan'])
projection = lambda source: {key: source[key] for key in ('source_id', 'columns', 'rows', 'row_ids')}
evidence = {'a_source': projection(opened['source']), 'a_attempt': opened['committed']['attempt'],
            'history_count': len(opened['history']), 'opened_current': current_result(opened), 'before': before}
reused = run_project(set_source(opened, ingest_csv(Path(sys.argv[2]).read_bytes())))
evidence.update(b_source=projection(reused['source']), b_attempt=reused['committed']['attempt'],
                after=plan_digest(reused['draft_plan']), reused_current=current_result(reused))
assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
print(json.dumps(evidence, allow_nan=False))
"""
        for include_history in (True, False):
            with self.subTest(include_history=include_history):
                payload = {"mode": "fulfillment-only", "project_name": "Acceptance roundtrip",
                           "fulfillment": export_project(project, include_history=include_history)}
                with tempfile.TemporaryDirectory(prefix="fulfillment-archive-") as directory:
                    archive_path = Path(directory) / "month-a.dataworks"
                    archive_path.write_bytes(dumps_project(payload))
                    completed = subprocess.run(
                        [sys.executable, "-c", code, str(archive_path), str(FIXTURES / "month_b.csv")],
                        cwd=ROOT, capture_output=True, text=True, timeout=30,
                    )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                child = json.loads(completed.stdout)
                self.assertEqual(child["a_attempt"], saved_attempt)
                self.assertTrue(child["opened_current"])
                self.assertTrue(child["reused_current"])
                self.assertEqual(child["before"], saved_digest)
                self.assertEqual(child["after"], saved_digest)
                self.assertEqual(child["history_count"] > 0, include_history)
                self.assert_completed(child["a_source"], child["a_attempt"], EXPECTED["completed_runs"]["month_a.csv"])
                self.assert_completed(child["b_source"], child["b_attempt"], EXPECTED["completed_runs"]["month_b.csv"])
                self.assertNotEqual(child["a_attempt"]["run_id"], child["b_attempt"]["run_id"])

    def test_execution_owns_results_and_never_mutates_source_or_plan(self):
        source = self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        plan = self.engine.default_plan()
        before_source, before_plan = copy.deepcopy(source), copy.deepcopy(plan)
        result = self.engine.execute(source, plan)
        self.assertEqual(source, before_source)
        self.assertEqual(plan, before_plan)
        result["accepted"][0]["values"][0] = "mutated-result"
        result["rejected"][0]["raw"]["order_id"] = "mutated-rejection"
        self.assertEqual(source, before_source)
        self.assertEqual(plan, before_plan)

    def test_repeated_attempts_have_unique_run_ids_and_stable_source_row_ids(self):
        data = (FIXTURES / "month_a.csv").read_bytes()
        first_source = self.engine.ingest_csv(data)
        second_source = self.engine.ingest_csv(data)
        self.assertEqual(first_source["source_id"], second_source["source_id"])
        self.assertEqual(first_source["row_ids"], second_source["row_ids"])
        self.assertEqual(len(set(first_source["row_ids"])), len(first_source["rows"]))
        plan = self.engine.default_plan()
        first, second = self.engine.execute(first_source, plan), self.engine.execute(second_source, plan)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["accepted"], second["accepted"])
        self.assertEqual(first["rejected"], second["rejected"])
        self.assertEqual(first["plan_digest"], self.engine.plan_digest(plan))

    def test_f04_permuted_columns_produce_the_same_canonical_values(self):
        text = (FIXTURES / "month_b.csv").read_text(encoding="utf-8")
        source_rows = list(csv.DictReader(io.StringIO(text)))
        columns = list(reversed(COLUMNS))
        data = csv_bytes([[row[column] for column in columns] for row in source_rows], columns)
        source, result = self.run_bytes(data)
        self.assert_completed(source, result, EXPECTED["completed_runs"]["month_b.csv"])

    def test_quantity_parsing_preserves_integer_precision_and_never_fills(self):
        quantities = [" +0002 ", "9223372036854775807", "9223372036854775808",
                      "-9223372036854775809", "-9223372036854775808", "1.0", "NaN", "1e3", "   "]
        _, result = self.run_bytes(csv_bytes([order_row(index, quantity=value) for index, value in enumerate(quantities, 1)]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["values"][4] for item in result["accepted"]], [2, 2**63 - 1])
        self.assertTrue(all(type(item["values"][4]) is int for item in result["accepted"]))
        self.assertEqual(result["report"]["accepted_quantity"], 2**63 + 1)
        self.assertEqual([item["reasons"] for item in result["rejected"]], [
            ["quantity_out_of_range"], ["quantity_out_of_range"], ["nonpositive_quantity"],
            ["invalid_quantity"], ["invalid_quantity"], ["invalid_quantity"], ["missing_quantity"],
        ])

    def test_date_parsing_is_explicit_calendar_checked_and_does_not_guess(self):
        dates = ["2024-02-29", "2026/10/03", "2026-02-29", "10/03/2026",
                 "2026-1-03", "2026/10-03", "2026-10-03T00:00:00", " 2026-10-03 ", ""]
        _, result = self.run_bytes(csv_bytes([order_row(index, date=value) for index, value in enumerate(dates, 1)]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["values"][2] for item in result["accepted"]], ["2024-02-29", "2026-10-03"])
        self.assertEqual([item["reasons"] for item in result["rejected"]], [["invalid_order_date"]] * 6 + [["missing_order_date"]])

    def test_unknown_status_prevents_commit_even_when_the_row_is_otherwise_invalid(self):
        _, result = self.run_bytes(csv_bytes([order_row(1, quantity="0", status=" Backordered ", sku="")]))
        self.assert_noncommitting(result, "decision_required")
        issue = next(issue for issue in result["issues"] if issue["code"] == "unseen_fulfillment_status")
        self.assertEqual(issue["source_records"], [1])
        self.assertEqual(issue["normalized_value"], "backordered")

    def test_unseen_status_is_resolved_only_by_an_explicit_versioned_plan_change(self):
        source = self.engine.ingest_csv((FIXTURES / "month_c_unseen_status.csv").read_bytes())
        original_plan = self.engine.default_plan()
        blocked = self.engine.execute(source, original_plan)
        self.assert_noncommitting(blocked, "decision_required")
        mapping = copy.deepcopy(original_plan["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        revised_plan = self.engine.revise_plan(original_plan, status_mapping=mapping)
        completed = self.engine.execute(source, revised_plan)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["accepted"][0]["values"][5], "pending")
        self.assertEqual(blocked["source_id"], completed["source_id"])
        self.assertNotEqual(blocked["plan_digest"], completed["plan_digest"])
        self.assertNotIn("backordered", original_plan["steps"][1]["status_mapping"])

    def test_f10_internal_failure_discards_candidate_outputs_and_preserves_inputs(self):
        source = self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        plan = self.engine.default_plan()
        before_source, before_plan = copy.deepcopy(source), copy.deepcopy(plan)
        canonicalize = self.engine._canonicalize_row
        calls = []

        def fail_later(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("injected late failure with source-private content")
            return canonicalize(*args, **kwargs)

        with patch.object(self.engine, "_canonicalize_row", side_effect=fail_later):
            result = self.engine.execute(source, plan)
        self.assertGreaterEqual(len(calls), 2)
        self.assert_noncommitting(result, "failed")
        issue = next(issue for issue in result["issues"] if issue["code"] == "execution_failed")
        self.assertEqual(issue["error_type"], "RuntimeError")
        self.assertNotIn("source-private content", json.dumps(result))
        self.assertEqual(source, before_source)
        self.assertEqual(plan, before_plan)

    def test_conflicting_key_blocks_the_whole_group_without_choosing_a_winner(self):
        rows = [order_row(1, quantity="1"), order_row(1, quantity="1"), order_row(1, quantity="2")]
        _, result = self.run_bytes(csv_bytes(rows))
        self.assert_noncommitting(result, "decision_required")
        issue = next(issue for issue in result["issues"] if issue["code"] == "conflicting_duplicate")
        self.assertEqual(issue["source_records"], [1, 2, 3])
        self.assertEqual(issue["differing_columns"], ["quantity"])

    def test_invalid_rows_are_quarantined_before_duplicate_conflict_classification(self):
        source, result = self.run_bytes(csv_bytes([order_row(1), order_row(1, quantity="-1")]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(result["rejected"][0]["reasons"], ["nonpositive_quantity"])
        self.assertEqual(result["accepted"][0]["row_id"], source["row_ids"][0])

    def test_identifier_tokens_are_preserved_and_whitespace_is_not_silently_trimmed(self):
        rows = [order_row(1), order_row(2), order_row(3)]
        rows[0][:2] = ["NA", "NULL"]
        rows[1][0] = " 000002"
        rows[2][1] = "01 "
        _, result = self.run_bytes(csv_bytes(rows))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["accepted"][0]["values"][:2], ["NA", "NULL"])
        self.assertEqual([item["reasons"] for item in result["rejected"]], [["identifier_whitespace"], ["identifier_whitespace"]])

    def test_source_identity_uses_parsed_records_rather_than_physical_csv_lines(self):
        rows = [order_row(1, sku=" multi\nline "), order_row(2)]
        source, result = self.run_bytes(csv_bytes(rows))
        self.assertEqual(len(source["row_ids"]), 2)
        self.assertEqual([item["source_record"] for item in result["accepted"]], [1, 2])
        self.assertEqual(result["accepted"][0]["values"][3], "MULTI\nLINE")

    def test_zero_rows_and_all_rejected_rows_remain_completed_reconciled_runs(self):
        for rows, rejected_count in (([], 0), ([order_row(1, quantity="0")], 1)):
            with self.subTest(rows=len(rows)):
                source, result = self.run_bytes(csv_bytes(rows))
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["accepted"], [])
                self.assertEqual(len(result["rejected"]), rejected_count)
                self.assertEqual(result["report"]["source_rows"], len(source["rows"]))
                self.assertEqual(result["report"]["accepted_rows"] + result["report"]["rejected_rows"], len(source["rows"]))


if __name__ == "__main__":
    unittest.main()
