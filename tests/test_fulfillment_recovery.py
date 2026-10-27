"""Cross-file recovery and archive integrity beyond isolated operation tests.

Inputs/expectations are original synthetic scenarios or the frozen Phase 1 oracle.
Temporary archives stay under the repository runner's disposable temp directory.
"""

import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from backend import fulfillment as engine
from backend import fulfillment_state as workspace
from backend.project_io import dumps_project, loads_project
from backend.state import StateManager


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "fulfillment_v1"
ORACLE = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def source(name="month_a.csv"):
    return engine.ingest_csv((FIXTURES / name).read_bytes())


def completed(name="month_a.csv"):
    return workspace.run_project(workspace.set_source(workspace.new_project(), source(name)))


def csv_source(rows, columns=engine.COLUMNS):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return engine.ingest_csv(output.getvalue().encode("utf-8"))


def archive(project, *, include_history=True):
    return dumps_project({"mode": "fulfillment-only", "project_name": "Recovery proof",
                          "fulfillment": workspace.export_project(project, include_history=include_history)})


FRESH_REOPEN = r"""
import json, sys
from pathlib import Path
from backend.fulfillment import plan_digest
from backend.fulfillment_state import validate_project, current_result, undo_project
from backend.project_io import loads_project
project = validate_project(loads_project(Path(sys.argv[1]).read_bytes())['fulfillment'])
original = project
undo_states = []
while project['history']:
    project = undo_project(project)
    attempt = project['last_attempt']
    undo_states.append({
        'source_id': project['source']['source_id'] if project['source'] else None,
        'plan_digest': plan_digest(project['draft_plan']),
        'latest_status': attempt['status'] if attempt else None,
        'committed_run_id': project['committed']['attempt']['run_id'] if project['committed'] else None,
        'current': current_result(project),
    })
assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
print(json.dumps({
    'current': current_result(original),
    'source_id': original['source']['source_id'],
    'plan_digest': plan_digest(original['draft_plan']),
    'plan_revision': original['draft_plan']['revision'],
    'latest_attempt': original['last_attempt'],
    'committed_attempt': original['committed']['attempt'],
    'raw_sha256': original['committed']['source']['raw_sha256'],
    'history_count': len(original['history']),
    'undo_states': undo_states,
}))
"""


class FulfillmentRecoveryTests(unittest.TestCase):
    def assert_prior_commit(self, project, expected, *, current=False):
        self.assertEqual(project["committed"], expected["committed"])
        self.assertEqual(workspace.current_result(project), current)

    def test_interleaved_plan_source_block_failure_and_undo_survive_fresh_reopen(self):
        initial = completed()
        b = workspace.run_project(workspace.set_source(initial, source("month_b.csv")))
        self.assertEqual(b["committed"]["attempt"]["report"], ORACLE["completed_runs"]["month_b.csv"]["report"])
        mapping = copy.deepcopy(b["draft_plan"]["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        revised_plan = engine.revise_plan(b["draft_plan"], status_mapping=mapping)
        revised_b = workspace.set_plan(b, revised_plan)
        self.assert_prior_commit(revised_b, b)
        drafted_c = workspace.set_source(revised_b, source("month_c_missing_quantity.csv"))
        blocked = workspace.run_project(drafted_c)
        self.assertEqual(blocked["last_attempt"]["status"], "blocked")
        self.assert_prior_commit(blocked, b)
        self.assertEqual(len(blocked["history"]), len(drafted_c["history"]))
        recovered_draft = workspace.undo_project(blocked)
        self.assertEqual(recovered_draft, revised_b)
        new_b = workspace.run_project(recovered_draft)
        self.assertTrue(workspace.current_result(new_b))
        self.assertNotEqual(new_b["committed"]["attempt"]["run_id"], b["committed"]["attempt"]["run_id"])
        with patch("backend.fulfillment._report", side_effect=RuntimeError("source-private-value")):
            failed = workspace.run_project(new_b)
        self.assert_prior_commit(failed, new_b, current=True)
        self.assertEqual(failed["last_attempt"]["status"], "failed")
        self.assertNotEqual(failed["last_attempt"]["run_id"], failed["committed"]["attempt"]["run_id"])
        self.assertEqual(failed["history"], new_b["history"])
        with tempfile.TemporaryDirectory(prefix="fulfillment-recovery-") as directory:
            for include_history in (True, False):
                with self.subTest(include_history=include_history):
                    path = Path(directory) / f"recovery-{include_history}.dataworks"
                    path.write_bytes(archive(failed, include_history=include_history))
                    result = subprocess.run([sys.executable, "-c", FRESH_REOPEN, str(path)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    reopened = json.loads(result.stdout)
                    self.assertTrue(reopened["current"])
                    self.assertEqual(reopened["source_id"], new_b["source"]["source_id"])
                    self.assertEqual(reopened["latest_attempt"], failed["last_attempt"])
                    self.assertEqual(reopened["committed_attempt"], new_b["committed"]["attempt"])
                    self.assertEqual(reopened["plan_digest"], engine.plan_digest(revised_plan))
                    self.assertEqual(reopened["plan_revision"], revised_plan["revision"])
                    self.assertEqual(reopened["raw_sha256"], source("month_b.csv")["raw_sha256"])
                    self.assertEqual(bool(reopened["history_count"]), include_history)
                    self.assertEqual(len(reopened["undo_states"]), reopened["history_count"])
                    if include_history:
                        self.assertTrue(any(item["committed_run_id"] == initial["committed"]["attempt"]["run_id"]
                                            for item in reopened["undo_states"]))
                    else:
                        self.assertEqual(reopened["undo_states"], [])
        self.assertNotIn("source-private-value", json.dumps(failed["last_attempt"]))

    def test_unknown_status_plan_resolution_and_undo_preserve_the_original_decision(self):
        initial = completed()
        drafted = workspace.set_source(initial, source("month_c_unseen_status.csv"))
        decision = workspace.run_project(drafted)
        self.assertEqual(decision["last_attempt"]["status"], "decision_required")
        self.assert_prior_commit(decision, initial)
        mapping = copy.deepcopy(decision["draft_plan"]["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        revised = workspace.set_plan(decision, engine.revise_plan(decision["draft_plan"], status_mapping=mapping))
        committed = workspace.run_project(revised)
        self.assertEqual(committed["committed"]["attempt"]["accepted"][0]["values"],
                         ["000501", "01", "2026-10-01", "SKU-401", 1, "pending"])
        self.assertEqual(committed["source"]["source_id"], decision["source"]["source_id"])
        self.assertNotEqual(committed["last_attempt"]["plan_digest"], decision["last_attempt"]["plan_digest"])
        reopened = workspace.validate_project(loads_project(archive(committed))["fulfillment"])
        restored_draft = workspace.undo_project(reopened)
        self.assertEqual(restored_draft, revised)
        restored_decision = workspace.undo_project(restored_draft)
        self.assertEqual(restored_decision, decision)
        self.assertEqual(restored_decision["last_attempt"]["issues"][0]["normalized_value"], "backordered")
        self.assertNotIn("backordered", restored_decision["draft_plan"]["steps"][1]["status_mapping"])

    def test_renamed_column_reports_missing_and_extra_without_replacing_prior_output(self):
        initial = completed("month_b.csv")
        columns = ["qty" if column == "quantity" else column for column in engine.COLUMNS]
        drifted = csv_source([["009001", "01", "2026-10-01", "sku-one", "2", "shipped"]], columns)
        drafted = workspace.set_source(initial, drifted)
        result = workspace.run_project(drafted)
        self.assertEqual(result["last_attempt"]["status"], "blocked")
        self.assertEqual({issue["code"]: issue["columns"] for issue in result["last_attempt"]["issues"]},
                         {"missing_required_column": ["quantity"], "unexpected_column": ["qty"]})
        self.assert_prior_commit(result, initial)
        reopened = workspace.validate_project(loads_project(archive(result))["fulfillment"])
        self.assertEqual(workspace.undo_project(reopened), initial)

    def test_changed_quantity_types_quarantine_rows_instead_of_guessing_a_type_conversion(self):
        initial = completed()
        rows = [
            ["009001", "01", "2026-10-01", "sku-one", "  +0002  ", "shipped"],
            ["009002", "01", "2026-10-02", "sku-two", "2.0", "pending"],
            ["009003", "01", "2026-10-03", "sku-three", "2e0", "pending"],
            ["009004", "01", "2026-10-04", "sku-four", "Infinity", "pending"],
            ["009005", "01", "2026-10-05", "sku-five", "", "cancelled"],
        ]
        updated = workspace.run_project(workspace.set_source(initial, csv_source(rows)))
        result = updated["committed"]["attempt"]
        self.assertEqual([record["values"] for record in result["accepted"]],
                         [["009001", "01", "2026-10-01", "SKU-ONE", 2, "fulfilled"]])
        self.assertEqual([record["reasons"] for record in result["rejected"]],
                         [["invalid_quantity"], ["invalid_quantity"], ["invalid_quantity"], ["missing_quantity"]])
        self.assertEqual(result["report"]["source_rows"], 5)
        self.assertEqual(result["report"]["accepted_quantity"], 2)
        reopened = workspace.validate_project(loads_project(archive(updated))["fulfillment"])
        self.assertEqual(reopened["committed"]["attempt"], result)
        self.assertEqual(workspace.undo_project(workspace.undo_project(reopened)), initial)

    def test_normalized_duplicate_counts_distinguish_lines_orders_and_quantities(self):
        rows = [
            ["000900", "01", "2026/10/02", " sku-x ", "+0002", " SHIPPED "],
            ["000900", "01", "2026-10-02", "SKU-X", "2", "fulfilled"],
            ["000900", "02", "2026-10-02", "sku-y", "1", "pending"],
            ["000901", "01", "2026-10-03", "sku-z", "3", "cancelled"],
        ]
        project = workspace.run_project(workspace.set_source(workspace.new_project(), csv_source(rows)))
        result = project["committed"]["attempt"]
        self.assertEqual(result["report"], {
            "source_rows": 4, "accepted_rows": 3, "rejected_rows": 1, "rejection_reason_assignments": 1,
            "accepted_distinct_orders": 2, "accepted_quantity": 6,
            "accepted_line_counts_by_status": {"fulfilled": 1, "pending": 1, "cancelled": 1},
            "accepted_quantity_by_status": {"fulfilled": 2, "pending": 1, "cancelled": 3},
        })
        self.assertEqual([item["source_record"] for item in result["accepted"]], [1, 3, 4])
        self.assertEqual(result["rejected"][0]["duplicate_of_source_record"], 1)
        self.assertEqual(result["rejected"][0]["raw"]["quantity"], "2")
        self.assertEqual(project["source"]["rows"][0][4], "+0002")
        self.assertEqual(workspace.validate_project(loads_project(archive(project))["fulfillment"]), project)

    def test_read_option_changes_make_identical_bytes_a_distinct_current_source(self):
        initial = completed()
        same_bytes = engine.ingest_csv(initial["source"]["raw_bytes"], encoding="utf-8")
        self.assertEqual(same_bytes["raw_sha256"], initial["source"]["raw_sha256"])
        self.assertNotEqual(same_bytes["source_id"], initial["source"]["source_id"])
        drafted = workspace.set_source(initial, same_bytes)
        self.assert_prior_commit(drafted, initial)
        rerun = workspace.run_project(drafted)
        self.assertEqual(rerun["committed"]["attempt"]["report"], initial["committed"]["attempt"]["report"])
        self.assertFalse(set(item["row_id"] for item in rerun["committed"]["attempt"]["accepted"])
                         & set(item["row_id"] for item in initial["committed"]["attempt"]["accepted"]))
        self.assertEqual(workspace.validate_project(loads_project(archive(rerun))["fulfillment"]), rerun)

    def test_mid_transform_failure_then_undo_recovers_the_entire_previous_source_result(self):
        initial = completed()
        drafted = workspace.set_source(initial, source("month_b.csv"))
        original = engine._canonicalize_row
        calls = []

        def fail_on_second(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("source-private-content")
            return original(*args, **kwargs)

        with patch("backend.fulfillment._canonicalize_row", side_effect=fail_on_second):
            failed = workspace.run_project(drafted)
        self.assertEqual(len(calls), 2)
        self.assert_prior_commit(failed, initial)
        self.assertEqual(failed["last_attempt"]["status"], "failed")
        self.assertIsNone(failed["last_attempt"]["accepted"])
        self.assertIsNone(failed["last_attempt"]["rejected"])
        self.assertIsNone(failed["last_attempt"]["report"])
        self.assertEqual(failed["last_attempt"]["issues"][0]["stage"], "validate_rows")
        reopened = workspace.validate_project(loads_project(archive(failed))["fulfillment"])
        self.assertEqual(workspace.undo_project(reopened), initial)
        self.assertEqual(initial["source"], source())

    def test_failed_attempt_cannot_reuse_a_successful_run_identifier_in_an_archive(self):
        initial = completed()
        with patch("backend.fulfillment._report", side_effect=RuntimeError("injected")):
            failed = workspace.run_project(initial)
        failed["last_attempt"]["run_id"] = failed["committed"]["attempt"]["run_id"]
        with self.assertRaises(ValueError):
            workspace.validate_project(failed)

    def test_completed_run_identifier_cannot_be_reassigned_to_another_historical_source(self):
        initial = completed()
        later = workspace.run_project(workspace.set_source(initial, source("month_b.csv")))
        original_run_id = initial["committed"]["attempt"]["run_id"]
        # The current B snapshot is internally coherent; the UUID collision is
        # only exposed by its different A meaning retained in prior checkpoints.
        later["last_attempt"]["run_id"] = original_run_id
        later["committed"]["attempt"]["run_id"] = original_run_id
        with self.assertRaises(ValueError):
            workspace.validate_project(later)

    def test_plan_edits_cannot_reuse_a_known_revision_for_different_semantics(self):
        initial = completed()
        before = copy.deepcopy(initial)
        edited_without_revision = engine.validate_plan(initial["draft_plan"])
        edited_without_revision["steps"][1]["status_mapping"]["backordered"] = "pending"
        self.assertNotEqual(engine.plan_digest(edited_without_revision), engine.plan_digest(initial["draft_plan"]))
        with self.assertRaises(ValueError):
            workspace.set_plan(initial, edited_without_revision)
        self.assertEqual(initial, before)
        forged = copy.deepcopy(initial)
        forged["draft_plan"] = edited_without_revision
        forged["last_attempt"] = None
        with self.assertRaises(ValueError):
            workspace.validate_project(forged)

    def test_conflicting_plan_revision_in_an_unexecuted_checkpoint_is_rejected(self):
        project = completed()
        self.assertIsNone(project["history"][0]["source"])
        self.assertIsNone(project["history"][0]["last_attempt"])
        self.assertIsNone(project["history"][0]["committed"])
        project["history"][0]["draft_plan"]["steps"][1]["status_mapping"]["backordered"] = "pending"
        with self.assertRaises(ValueError):
            workspace.validate_project(project)

    def test_importing_an_earlier_known_revision_preserves_its_original_meaning(self):
        initial = completed()
        earlier = copy.deepcopy(initial["draft_plan"])
        mapping = copy.deepcopy(earlier["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        revised = workspace.set_plan(initial, engine.revise_plan(earlier, status_mapping=mapping))
        self.assertFalse(workspace.current_result(revised))
        imported = workspace.set_plan(revised, engine.loads_plan(engine.dumps_plan(earlier)))
        self.assertEqual(imported["draft_plan"], earlier)
        self.assertTrue(workspace.current_result(imported))
        self.assertEqual(imported["committed"], initial["committed"])
        self.assertIsNone(imported["last_attempt"])
        self.assertEqual(workspace.validate_project(loads_project(archive(imported))["fulfillment"]), imported)

    def test_validation_caches_do_not_alias_current_committed_or_historical_evidence(self):
        original = completed()
        validated = workspace.validate_project(original)
        current_source = validated["source"]
        committed_source = validated["committed"]["source"]
        historical_source = validated["history"][-1]["source"]
        self.assertIsNot(current_source, committed_source)
        self.assertIsNot(current_source["rows"], committed_source["rows"])
        self.assertIsNot(current_source["rows"], historical_source["rows"])
        current_source["rows"][0][0] = "changed-current-source"
        self.assertEqual(committed_source["rows"][0][0], "000101")
        self.assertEqual(historical_source["rows"][0][0], "000101")
        self.assertEqual(original["source"]["rows"][0][0], "000101")
        validated["last_attempt"]["accepted"][0]["values"][0] = "changed-latest-result"
        self.assertEqual(validated["committed"]["attempt"]["accepted"][0]["values"][0], "000101")
        self.assertEqual(original["last_attempt"]["accepted"][0]["values"][0], "000101")
        validated["draft_plan"]["steps"][1]["status_mapping"]["open"] = "cancelled"
        self.assertEqual(validated["committed"]["plan"]["steps"][1]["status_mapping"]["open"], "pending")
        self.assertEqual(validated["history"][-1]["draft_plan"]["steps"][1]["status_mapping"]["open"], "pending")
        self.assertEqual(original["draft_plan"]["steps"][1]["status_mapping"]["open"], "pending")


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FulfillmentAtomicRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        patcher = patch("backend.state.st.session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        StateManager.init_session()
        StateManager.load_dataset(pd.DataFrame({"unrelated": ["keep-me"]}), {})
        StateManager.set_fulfillment_source(source())
        StateManager.run_fulfillment()
        StateManager.set_project_name("Previously verified project")

    def assert_rejected_atomically(self, payload):
        fulfillment = copy.deepcopy(self.state.fulfillment)
        frame = self.state.df.copy(deep=True)
        revision, name = self.state.project_revision, self.state.project_name
        prepared = {"data": b"prior prepared archive", "token": (revision, True)}
        self.state.prepared_project_download = prepared
        with self.assertRaises(ValueError):
            StateManager.restore_project(loads_project(dumps_project(payload)))
        self.assertEqual(self.state.fulfillment, fulfillment)
        assert_frame_equal(self.state.df, frame)
        self.assertEqual(self.state.project_revision, revision)
        self.assertEqual(self.state.project_name, name)
        self.assertIs(self.state.prepared_project_download, prepared)

    def test_corrupt_dtype_order_duplicate_reference_source_and_report_do_not_replace_active_data(self):
        valid = StateManager.export_project()
        mutations = [
            lambda item: item["committed"]["attempt"]["accepted"][0]["values"].__setitem__(4, 2.0),
            lambda item: item["committed"]["attempt"]["accepted"].reverse(),
            lambda item: item["committed"]["attempt"]["rejected"][0].update(duplicate_of_source_record=1,
                duplicate_of_row_id=item["committed"]["source"]["row_ids"][0]),
            lambda item: item["committed"].update(source=source("month_b.csv")),
            lambda item: item["committed"]["attempt"]["report"]["accepted_quantity_by_status"].update(fulfilled=4, pending=0),
            lambda item: item["history"][-1]["source"]["rows"].reverse(),
        ]
        for index, mutate in enumerate(mutations):
            corrupt = copy.deepcopy(valid)
            mutate(corrupt["fulfillment"])
            with self.subTest(mutation=index):
                self.assert_rejected_atomically(corrupt)

    def test_canonical_value_forgery_with_consistent_totals_is_rejected_by_reexecution(self):
        corrupt = StateManager.export_project()
        # Both lines keep the same order count, quantities, row partition and
        # logical types. Only replay against original evidence exposes the lie.
        for attempt in (corrupt["fulfillment"]["last_attempt"], corrupt["fulfillment"]["committed"]["attempt"]):
            attempt["accepted"][0]["values"][0] = "999999"
            attempt["accepted"][1]["values"][0] = "999999"
        self.assert_rejected_atomically(corrupt)

    def test_stale_source_plan_and_committed_run_associations_are_rejected(self):
        valid = StateManager.export_project()
        variants = []
        corrupt = copy.deepcopy(valid)
        corrupt["fulfillment"]["source"] = source("month_b.csv")
        variants.append(corrupt)
        corrupt = copy.deepcopy(valid)
        mapping = copy.deepcopy(corrupt["fulfillment"]["draft_plan"]["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        corrupt["fulfillment"]["draft_plan"] = engine.revise_plan(corrupt["fulfillment"]["draft_plan"], status_mapping=mapping)
        variants.append(corrupt)
        corrupt = copy.deepcopy(valid)
        corrupt["fulfillment"]["committed"]["attempt"]["run_id"] = "1" * 32
        variants.append(corrupt)
        for index, corrupt in enumerate(variants):
            with self.subTest(variant=index):
                self.assert_rejected_atomically(corrupt)

    def test_unsupported_statistical_policy_and_recipe_steps_never_enter_the_workflow(self):
        for policy in ("refit", "frozen"):
            corrupt = StateManager.export_project()
            corrupt["fulfillment"]["draft_plan"]["statistical_policy"] = policy
            with self.subTest(policy=policy):
                self.assert_rejected_atomically(corrupt)
        corrupt = StateManager.export_project()
        corrupt["fulfillment"]["draft_plan"]["steps"].append(
            {"kind": "missing", "version": 1, "strategies": {"quantity": "median"}})
        self.assert_rejected_atomically(corrupt)

    def test_malformed_replacement_source_leaves_active_source_commit_and_archive_intact(self):
        before = copy.deepcopy(self.state.fulfillment)
        revision = self.state.project_revision
        self.state.prepared_project_download = {"data": b"retained"}
        for raw in (b'order_id,line_id\n"unterminated,01\n', b"order_id,order_id\n1,2\n",
                    b"order_id,line_id\n1,2,3\n", b"order_id,line_id\n\xff,01\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                StateManager.set_fulfillment_source(engine.ingest_csv(raw))
            self.assertEqual(self.state.fulfillment, before)
            self.assertEqual(self.state.project_revision, revision)
            self.assertEqual(self.state.prepared_project_download["data"], b"retained")


if __name__ == "__main__":
    unittest.main()
