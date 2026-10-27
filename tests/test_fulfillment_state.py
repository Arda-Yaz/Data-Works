"""Fulfillment commitment/recovery checks against independent source fixtures."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
import plotly.graph_objects as go

from backend import fulfillment, fulfillment_state, resources
from backend.project_io import dumps_project, loads_project
from backend.state import StateManager


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"


def source(name="month_a.csv"):
    return fulfillment.ingest_csv((FIXTURES / name).read_bytes())


def completed_project():
    return fulfillment_state.run_project(fulfillment_state.set_source(fulfillment_state.new_project(), source()))


class FulfillmentStateTests(unittest.TestCase):
    def test_transitions_own_inputs_and_new_projects_are_independent(self):
        first, second = fulfillment_state.new_project(), fulfillment_state.new_project()
        self.assertIsNot(first["draft_plan"]["steps"], second["draft_plan"]["steps"])
        before = copy.deepcopy(first)
        uploaded = source()
        candidate = fulfillment_state.set_source(first, uploaded)
        self.assertEqual(first, before)
        uploaded["rows"][0][0] = "changed"
        self.assertEqual(candidate["source"]["rows"][0][0], "000101")
        candidate["history"][0]["draft_plan"]["revision"] += 1
        self.assertEqual(first["draft_plan"]["revision"], 1)

    def test_successful_commit_owns_exact_source_plan_and_attempt(self):
        project = completed_project()
        self.assertEqual(project["last_attempt"]["status"], "completed")
        self.assertEqual(project["committed"]["attempt"]["report"]["accepted_rows"], 4)
        self.assertTrue(fulfillment_state.current_result(project))
        self.assertEqual(fulfillment_state.validate_project(project), project)
        project["last_attempt"]["accepted"][0]["values"][0] = "changed"
        self.assertEqual(project["committed"]["attempt"]["accepted"][0]["values"][0], "000101")

    def test_blocked_and_decision_attempts_preserve_prior_commit_without_checkpoint(self):
        original = completed_project()
        for name, status in (("month_c_missing_quantity.csv", "blocked"),
                             ("month_c_extra_column.csv", "blocked"),
                             ("month_c_conflicting_duplicates.csv", "decision_required"),
                             ("month_c_unseen_status.csv", "decision_required")):
            with self.subTest(input=name):
                drafted = fulfillment_state.set_source(original, source(name))
                history = copy.deepcopy(drafted["history"])
                result = fulfillment_state.run_project(drafted)
                self.assertEqual(result["last_attempt"]["status"], status)
                self.assertEqual(result["committed"], original["committed"])
                self.assertEqual(result["history"], history)
                self.assertIsNone(result["last_attempt"]["accepted"])
                self.assertFalse(fulfillment_state.current_result(result))
                self.assertEqual(drafted["last_attempt"], None)

    def test_late_failure_preserves_current_success_and_records_no_cell_message(self):
        project = completed_project()
        committed, history = copy.deepcopy(project["committed"]), copy.deepcopy(project["history"])
        with patch("backend.fulfillment._report", side_effect=RuntimeError("private-cell-value")):
            failed = fulfillment_state.run_project(project)
        self.assertEqual(failed["committed"], committed)
        self.assertEqual(failed["history"], history)
        self.assertEqual(failed["last_attempt"]["status"], "failed")
        self.assertEqual(failed["last_attempt"]["issues"][0]["stage"], "reconcile")
        self.assertEqual(failed["last_attempt"]["issues"][0]["error_type"], "RuntimeError")
        self.assertNotIn("private-cell-value", str(failed["last_attempt"]))
        self.assertTrue(fulfillment_state.current_result(failed))
        self.assertEqual(fulfillment_state.validate_project(failed), failed)

    def test_unexpected_engine_exception_is_recorded_without_overwriting_success(self):
        project = completed_project()
        with patch("backend.fulfillment_state.execute", side_effect=RuntimeError("private")):
            failed = fulfillment_state.run_project(project)
        self.assertEqual(failed["committed"], project["committed"])
        self.assertEqual(failed["last_attempt"]["status"], "failed")
        self.assertEqual(failed["last_attempt"]["issues"][0]["error_type"], "RuntimeError")

    def test_source_and_plan_edits_make_results_earlier_and_undo_restores_associations(self):
        project = completed_project()
        changed_source = fulfillment_state.set_source(project, source("month_b.csv"))
        self.assertFalse(fulfillment_state.current_result(changed_source))
        self.assertEqual(changed_source["committed"], project["committed"])
        self.assertIsNone(changed_source["last_attempt"])
        self.assertEqual(fulfillment_state.undo_project(changed_source), project)
        mapping = copy.deepcopy(project["draft_plan"]["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        changed_plan = fulfillment_state.set_plan(project, fulfillment.revise_plan(project["draft_plan"], status_mapping=mapping))
        self.assertFalse(fulfillment_state.current_result(changed_plan))
        self.assertEqual(changed_plan["committed"], project["committed"])
        self.assertEqual(fulfillment_state.undo_project(changed_plan), project)

    def test_month_b_success_and_undo_restore_the_whole_preceding_draft(self):
        drafted = fulfillment_state.set_source(completed_project(), source("month_b.csv"))
        committed = fulfillment_state.run_project(drafted)
        self.assertEqual(committed["committed"]["attempt"]["report"]["accepted_rows"], 5)
        self.assertEqual(fulfillment_state.undo_project(committed), drafted)
        self.assertFalse(fulfillment_state.current_result(drafted))

    def test_undo_has_count_and_memory_limits_without_discarding_current_commit(self):
        project = completed_project()
        for _ in range(13):
            plan = fulfillment.revise_plan(project["draft_plan"], status_mapping=project["draft_plan"]["steps"][1]["status_mapping"])
            project = fulfillment_state.set_plan(project, plan)
        self.assertEqual(len(project["history"]), 10)
        self.assertTrue(all("history" not in checkpoint for checkpoint in project["history"]))
        fresh = fulfillment_state.new_project()
        with patch.object(resources, "MAX_HISTORY_BYTES", 0):
            fresh = fulfillment_state.set_source(fresh, source())
            fresh = fulfillment_state.run_project(fresh)
            self.assertEqual(fresh["history"], [])
            self.assertEqual(fulfillment_state.undo_project(fresh), fresh)
            self.assertTrue(fulfillment_state.current_result(fresh))

    def test_export_without_history_keeps_current_evidence_and_does_not_mutate(self):
        project = completed_project()
        before = copy.deepcopy(project)
        exported = fulfillment_state.export_project(project, include_history=False)
        self.assertEqual(exported["history"], [])
        self.assertEqual(exported["committed"], project["committed"])
        self.assertEqual(project, before)
        reopened = loads_project(dumps_project({"fulfillment": exported}))["fulfillment"]
        self.assertEqual(fulfillment_state.validate_project(reopened), exported)
        self.assertEqual(fulfillment_state.undo_project(reopened), reopened)
        with self.assertRaises(ValueError):
            fulfillment_state.export_project(project, include_history="yes")

    def test_corrupted_types_outputs_and_associations_fail_validation(self):
        project = completed_project()
        mutations = [
            lambda item: item.update(version=True),
            lambda item: item.update(arbitrary_code="never execute"),
            lambda item: item["source"]["rows"][0].__setitem__(0, "tampered"),
            lambda item: item["committed"]["attempt"].update(run_id="invalid"),
            lambda item: item["committed"]["attempt"].update(engine_version=2),
            lambda item: item["committed"]["attempt"].update(plan_revision=True),
            lambda item: item["committed"]["attempt"]["report"].update(accepted_rows=4.0),
            lambda item: item["committed"]["attempt"]["accepted"][1]["values"].__setitem__(4, True),
            lambda item: item["committed"]["attempt"]["rejected"][0].update(duplicate_of_source_record=1),
            lambda item: item["committed"]["attempt"].update(source_id="0" * 64),
            lambda item: item["history"][0].update(history=[]),
        ]
        for index, mutate in enumerate(mutations):
            corrupt = copy.deepcopy(project)
            mutate(corrupt)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                fulfillment_state.validate_project(corrupt)

    def test_no_source_run_is_atomic(self):
        empty = fulfillment_state.new_project()
        before = copy.deepcopy(empty)
        with self.assertRaisesRegex(ValueError, "source"):
            fulfillment_state.run_project(empty)
        self.assertEqual(empty, before)

    def test_latest_completed_attempt_must_be_the_committed_success(self):
        initial = completed_project()
        later = fulfillment_state.run_project(fulfillment_state.set_source(initial, source("month_b.csv")))
        for committed in (None, initial["committed"]):
            corrupt = copy.deepcopy(later)
            corrupt["committed"] = copy.deepcopy(committed)
            with self.subTest(committed_present=committed is not None), self.assertRaisesRegex(ValueError, "committed"):
                fulfillment_state.validate_project(corrupt)
        corrupt = copy.deepcopy(later)
        corrupt["committed"]["attempt"]["run_id"] = "1" * 32
        with self.assertRaisesRegex(ValueError, "committed"):
            fulfillment_state.validate_project(corrupt)

    def test_failed_diagnostics_cannot_contain_messages_or_unsupported_execution_fields(self):
        project = completed_project()
        with patch("backend.fulfillment._report", side_effect=RuntimeError("private")):
            failed = fulfillment_state.run_project(project)
        for field, value in (("message", "private"), ("command", "never execute"), ("stage", "future-stage")):
            corrupt = copy.deepcopy(failed)
            corrupt["last_attempt"]["issues"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fulfillment_state.validate_project(corrupt)


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FulfillmentSessionTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        patcher = patch("backend.state.st.session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        StateManager.init_session()

    def complete(self):
        StateManager.set_fulfillment_source(source())
        return StateManager.run_fulfillment()

    def test_success_and_failed_rerun_invalidate_prepared_download_and_preserve_dataset(self):
        StateManager.load_dataset(pd.DataFrame({"unrelated": [1, 2]}), {})
        original = self.state.df.copy(deep=True)
        self.complete()
        committed = copy.deepcopy(self.state.fulfillment["committed"])
        revision = self.state.project_revision
        self.state.prepared_project_download = {"data": b"prior"}
        with patch("backend.fulfillment._report", side_effect=RuntimeError("private-cell")):
            result = StateManager.run_fulfillment()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.state.fulfillment["committed"], committed)
        self.assertGreater(self.state.project_revision, revision)
        self.assertNotIn("prepared_project_download", self.state)
        self.assertTrue(StateManager.fulfillment_result_is_current())
        assert_frame_equal(self.state.df, original)

    def test_noop_setters_do_not_invalidate_prepared_download(self):
        self.complete()
        self.state.prepared_project_download = {"data": b"retained"}
        revision = self.state.project_revision
        self.assertFalse(StateManager.set_fulfillment_source(copy.deepcopy(self.state.fulfillment["source"])))
        self.assertFalse(StateManager.set_fulfillment_plan(copy.deepcopy(self.state.fulfillment["draft_plan"])))
        self.assertEqual(self.state.project_revision, revision)
        self.assertIn("prepared_project_download", self.state)

    def test_fulfillment_only_archive_reopens_and_clears_generic_data(self):
        self.complete()
        StateManager.set_project_name("Monthly fulfillment")
        payload = StateManager.export_project()
        self.assertEqual(payload["mode"], "fulfillment-only")
        self.assertNotIn("df", payload)
        expected = copy.deepcopy(self.state.fulfillment)
        archive = dumps_project(payload)
        StateManager.load_dataset(pd.DataFrame({"unrelated": [4, 5]}), {})
        StateManager.restore_project(loads_project(archive))
        self.assertFalse(StateManager.has_dataset())
        self.assertEqual(self.state.fulfillment, expected)
        self.assertEqual(self.state.project_name, "Monthly fulfillment")
        self.assertTrue(StateManager.fulfillment_result_is_current())

    def test_mixed_archive_preserves_generic_data_and_legacy_resets_extension(self):
        StateManager.load_dataset(pd.DataFrame({"unrelated": [1, 2]}), {})
        self.complete()
        payload = StateManager.export_project()
        expected = copy.deepcopy(self.state.fulfillment)
        StateManager.clear()
        StateManager.restore_project(loads_project(dumps_project(payload)))
        self.assertEqual(self.state.fulfillment, expected)
        self.assertEqual(self.state.df["unrelated"].tolist(), [1, 2])
        del payload["fulfillment"]
        StateManager.restore_project(payload)
        self.assertIsNone(self.state.fulfillment["source"])
        self.assertIsNone(self.state.fulfillment["committed"])

    def test_invalid_extensions_and_modes_fail_before_any_active_state_change(self):
        self.complete()
        valid = StateManager.export_project()
        cases = []
        bad = copy.deepcopy(valid)
        bad["fulfillment"]["committed"]["attempt"]["report"]["accepted_rows"] = 99
        cases.append(bad)
        for field, value in (("mode", "future-mode"), ("project_name", ""), ("fulfillment", None)):
            bad = copy.deepcopy(valid)
            bad[field] = value
            cases.append(bad)
        StateManager.load_dataset(pd.DataFrame({"unrelated": [9]}), {})
        mixed = StateManager.export_project()
        mixed["fulfillment"] = None
        cases.append(mixed)
        for corrupt in cases:
            revision = self.state.project_revision
            before = StateManager.export_project()
            self.state.prepared_project_download = {"data": b"retained"}
            with self.subTest(fields=list(corrupt)), self.assertRaises(ValueError):
                StateManager.restore_project(corrupt)
            self.assertEqual(self.state.project_revision, revision)
            self.assertEqual(self.state.fulfillment, before["fulfillment"])
            assert_frame_equal(self.state.df, before["df"])
            self.assertEqual(self.state.prepared_project_download["data"], b"retained")

    def test_fulfillment_undo_and_optional_history_are_explicit(self):
        self.complete()
        StateManager.set_fulfillment_source(source("month_b.csv"))
        self.assertFalse(StateManager.fulfillment_result_is_current())
        self.assertTrue(StateManager.undo_fulfillment())
        self.assertTrue(StateManager.fulfillment_result_is_current())
        payload = StateManager.export_project(include_history=False)
        self.assertEqual(payload["fulfillment"]["history"], [])
        StateManager.restore_project(payload)
        self.assertFalse(StateManager.undo_fulfillment())

    def test_clear_dataset_preserves_fulfillment_work_and_project_name(self):
        StateManager.load_dataset(pd.DataFrame({"unrelated": [1, 1, 2]}), {})
        StateManager.apply_operation(
            "Remove generic duplicates", self.state.df.drop_duplicates(),
            operation={"kind": "duplicates", "parameters": {"subset": ["unrelated"]}},
        )
        StateManager.save_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])))
        StateManager.add_experiment({
            "dataset_id": self.state.dataset_id, "data_version": self.state.data_version,
            "config": {"source": "working"}, "results": {"test": {"score": 1}},
        })
        self.complete()
        StateManager.set_project_name("Retail fulfillment August")
        owned_fulfillment = self.state.fulfillment
        expected_fulfillment = copy.deepcopy(owned_fulfillment)
        self.assertTrue(self.state.history)
        self.assertTrue(self.state.viz_charts)
        self.assertTrue(self.state.experiments)
        revision = self.state.project_revision
        self.state.prepared_project_download = {"data": b"previous mixed project"}

        StateManager.clear_dataset()

        self.assertIsNone(self.state.df)
        self.assertIsNone(self.state.original_df)
        self.assertIsNone(self.state.dataset_id)
        self.assertIsNone(self.state.data_version)
        for field in ("history", "operation_log", "viz_charts", "experiments"):
            self.assertEqual(self.state[field], [], field)
        for field in ("column_meta", "original_column_meta", "cleaning_previews",
                      "_analysis_cache", "_visualization_cache", "_upload_summary_cache"):
            self.assertEqual(self.state[field], {}, field)
        self.assertIs(self.state.fulfillment, owned_fulfillment)
        self.assertEqual(self.state.fulfillment, expected_fulfillment)
        self.assertTrue(StateManager.fulfillment_result_is_current())
        self.assertEqual(self.state.project_name, "Retail fulfillment August")
        self.assertGreater(self.state.project_revision, revision)
        self.assertNotIn("prepared_project_download", self.state)
        self.assertEqual(StateManager.export_project()["mode"], "fulfillment-only")


if __name__ == "__main__":
    unittest.main()
