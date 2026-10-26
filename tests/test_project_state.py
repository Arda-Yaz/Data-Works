import copy
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.testing import assert_frame_equal

from backend.experiments import create_experiment
from backend.ingestion import infer_column_meta
from backend.ml import evaluate_model, get_available_models, train_model
from backend.operations import replay_operations
from backend.preprocessing import auto_preprocess
from backend.project_io import dumps_project, loads_project
from backend.state import MAX_EXPERIMENTS, MAX_HISTORY, StateManager


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class ProjectStateTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        patcher = patch("backend.state.st.session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        StateManager.init_session()
        frame = pd.DataFrame({
            "x": np.arange(40, dtype=float),
            "kind": ["a", "b"] * 20,
            "target": np.arange(40, dtype=float) * 0.3 + 0.1,
        })
        StateManager.load_dataset(frame, infer_column_meta(frame))

    def completed_experiment(self):
        train, test, y_train, y_test, info = auto_preprocess(
            self.state.df, "target", self.state.column_meta, problem_type="regression",
        )
        model = train_model(get_available_models("regression")["Linear Regression"], train, y_train)
        metrics = evaluate_model(model, test, y_test, "regression")
        metrics["_model"] = model
        return create_experiment(
            dataset_id=self.state.dataset_id, data_version=self.state.data_version,
            config={"source": "working", "target": "target", "features": ["x", "kind"],
                    "problem_type": "regression", "split_strategy": "random", "test_size": 0.2,
                    "group_col": None, "time_col": None, "random_state": 42},
            results={"Linear Regression": metrics}, errors={}, y_test=y_test,
            feature_names=train.columns.tolist(), feature_columns=info["feature_columns"],
            info=info, schema=self.state.column_meta,
        )

    def change_rules(self, allowed):
        metadata = copy.deepcopy(self.state.column_meta)
        metadata["kind"]["allowed_values"] = allowed
        return StateManager.apply_operation(
            "Update allowed values", self.state.df, column_meta=metadata,
            operation={"kind": "rules", "parameters": {"allowed_values": {"kind": allowed}}},
        )

    def test_project_round_trip_preserves_data_schema_history_charts_and_reports(self):
        record = self.completed_experiment()
        StateManager.add_experiment(record)
        self.change_rules(["a"])
        StateManager.set_target("target")
        StateManager.set_project_name("My analysis")
        StateManager.save_chart(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])))
        expected = StateManager.export_project()
        archived = dumps_project(expected)
        StateManager.clear()
        StateManager.restore_project(loads_project(archived))
        assert_frame_equal(self.state.df, expected["df"])
        assert_frame_equal(self.state.original_df, expected["original_df"])
        for key in ("column_meta", "original_column_meta", "operation_log", "data_version",
                    "dataset_id", "target_column", "schema_confirmed", "project_name"):
            self.assertEqual(self.state[key], expected[key], key)
        self.assertEqual(len(self.state.history), 1)
        self.assertEqual(len(self.state.viz_charts), 1)
        self.assertEqual(self.state.viz_charts[0].to_json(), go.Figure(expected["viz_charts"][0]).to_json())
        restored = self.state.experiments[0]
        self.assertEqual(restored["results"], expected["experiments"][0]["results"])
        self.assertNotIn("models", restored)
        self.assertNotIn("preprocessor", restored)
        self.assertTrue(StateManager.undo())
        self.assertIsNone(self.state.column_meta["kind"]["allowed_values"])

    def test_invalid_projects_are_rejected_before_active_state_changes(self):
        self.change_rules(["a"])
        StateManager.save_chart(go.Figure(go.Bar(x=["a"], y=[1])))
        StateManager.add_experiment(self.completed_experiment())
        valid = StateManager.export_project()
        invalids = []
        for field, value in (("dataset_id", "invalid"), ("schema_confirmed", "yes"),
                             ("target_column", "absent"), ("project_name", "")):
            project = copy.deepcopy(valid)
            project[field] = value
            invalids.append(project)
        project = copy.deepcopy(valid)
        project["column_meta"]["kind"]["role"] = "invalid"
        invalids.append(project)
        project = copy.deepcopy(valid)
        project["history"][0]["operation_count"] = 100
        invalids.append(project)
        project = copy.deepcopy(valid)
        project["experiments"][0]["results"]["Linear Regression"]["y_pred"] = []
        invalids.append(project)
        project = copy.deepcopy(valid)
        project["experiments"].append(copy.deepcopy(project["experiments"][0]))
        invalids.append(project)
        project = copy.deepcopy(valid)
        project["operation_log"][0]["effects"] = {"rows_before": -1}
        invalids.append(project)
        project = copy.deepcopy(valid)
        project["viz_charts"] = [{"invalid": "figure"}]
        invalids.append(project)
        for project in invalids:
            with self.subTest(fields=list(project)):
                previous = dumps_project(StateManager.export_project())
                revision = self.state.project_revision
                with self.assertRaises(ValueError):
                    StateManager.restore_project(project)
                self.assertEqual(dumps_project(StateManager.export_project()), previous)
                self.assertEqual(self.state.project_revision, revision)

    def test_undo_restores_data_version_and_truncates_recipe_beyond_snapshot_limit(self):
        original = self.state.original_df.copy()
        original_meta = copy.deepcopy(self.state.original_column_meta)
        versions = [self.state.data_version]
        for index in range(MAX_HISTORY + 3):
            self.change_rules(["a"] if index % 2 == 0 else ["b"])
            versions.append(self.state.data_version)
        self.assertEqual(len(self.state.history), MAX_HISTORY)
        self.assertEqual(len(self.state.operation_log), MAX_HISTORY + 3)
        StateManager.undo()
        self.assertEqual(self.state.data_version, versions[-2])
        self.assertEqual(len(self.state.operation_log), MAX_HISTORY + 2)
        replayed = replay_operations(original, original_meta, self.state.operation_log)
        assert_frame_equal(replayed["df"], self.state.df)
        self.assertEqual(replayed["column_meta"], self.state.column_meta)

    def test_noop_and_failed_operations_do_not_change_recipe_or_version(self):
        version, revision = self.state.data_version, self.state.project_revision
        self.assertFalse(StateManager.apply_operation("No change", self.state.df.copy()))
        with self.assertRaises(ValueError):
            StateManager.apply_operation("Invalid", self.state.df, operation={"kind": "unknown"})
        self.assertEqual(self.state.data_version, version)
        self.assertEqual(self.state.project_revision, revision)
        self.assertEqual(self.state.operation_log, [])

    def test_experiments_are_owned_bounded_and_reassociate_after_undo(self):
        record = self.completed_experiment()
        version = self.state.data_version
        identifiers = []
        for index in range(MAX_EXPERIMENTS + 2):
            record["id"] = uuid4().hex
            record["config"]["random_state"] = index
            identifiers.append(StateManager.add_experiment(record))
        self.assertEqual([item["id"] for item in self.state.experiments], identifiers[-MAX_EXPERIMENTS:])
        record["config"]["random_state"] = 100
        self.assertNotEqual(self.state.experiments[-1]["config"]["random_state"], 100)
        self.change_rules(["a"])
        self.assertNotEqual(self.state.data_version, version)
        self.assertEqual(self.state.experiments[-1]["data_version"], version)
        StateManager.undo()
        self.assertEqual(self.state.data_version, self.state.experiments[-1]["data_version"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            StateManager.add_experiment(self.state.experiments[-1])

    def test_replacing_or_clearing_dataset_removes_associated_experiments(self):
        StateManager.add_experiment(self.completed_experiment())
        previous_id = self.state.dataset_id
        StateManager.load_dataset(pd.DataFrame({"other": [1, 2]}), {})
        self.assertNotEqual(self.state.dataset_id, previous_id)
        self.assertEqual(self.state.experiments, [])
        self.assertEqual(self.state.operation_log, [])
        StateManager.clear()
        self.assertIsNone(self.state.dataset_id)
        self.assertIsNone(self.state.data_version)

    def test_export_can_omit_snapshots_without_losing_operation_recipe(self):
        self.change_rules(["a"])
        payload = loads_project(dumps_project(StateManager.export_project(include_history=False)))
        self.assertEqual(payload["history"], [])
        self.assertEqual(len(payload["operation_log"]), 1)
        StateManager.restore_project(payload)
        self.assertFalse(StateManager.undo())
        self.assertEqual(self.state.column_meta["kind"]["allowed_values"], ["a"])

    def test_legacy_session_is_honestly_non_replayable_and_can_be_saved(self):
        self.state.history = [{"name": "Old edit", "df": self.state.df.copy(),
                               "column_meta": copy.deepcopy(self.state.column_meta)}]
        for key in ("operation_log", "original_column_meta", "dataset_id", "data_version"):
            del self.state[key]
        StateManager.init_session()
        self.assertFalse(self.state.operation_log[0]["replayable"])
        StateManager.restore_project(loads_project(dumps_project(StateManager.export_project())))
        self.assertEqual(self.state.operation_log[0]["kind"], "legacy")
        self.assertTrue(StateManager.undo())


if __name__ == "__main__":
    unittest.main()
