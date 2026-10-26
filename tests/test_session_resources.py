import copy
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.testing import assert_frame_equal

from backend import resources
from backend.state import StateManager


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class SessionResourceTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        patcher = patch("backend.state.st.session_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        StateManager.init_session()
        StateManager.load_dataset(pd.DataFrame({"x": [1, 2, 3]}), {})

    def record(self, *, extra_bytes=0):
        return {
            "id": uuid4().hex, "dataset_id": self.state.dataset_id,
            "data_version": self.state.data_version,
            "config": {"selected_models": ["test"]}, "results": {"test": {"score": 1}},
            "buffer": np.zeros(extra_bytes, dtype=np.uint8),
        }

    def test_dataset_limit_rejects_load_and_operation_without_changing_state(self):
        frame = self.state.df.copy(deep=True)
        dataset_id, version = self.state.dataset_id, self.state.data_version
        with patch.object(resources, "MAX_DATASET_ROWS", 2):
            with self.assertRaises(ValueError):
                StateManager.load_dataset(pd.DataFrame({"x": [3, 4, 5]}), {})
            with self.assertRaises(ValueError):
                StateManager.apply_operation("Too many rows", self.state.df.assign(x=[2, 3, 4]))
        assert_frame_equal(self.state.df, frame)
        self.assertEqual(self.state.dataset_id, dataset_id)
        self.assertEqual(self.state.data_version, version)
        self.assertEqual(self.state.history, [])

    def test_history_budget_evicts_oldest_snapshots_but_keeps_full_journal(self):
        size = resources.estimate_bytes(StateManager._snapshot("Change"))
        versions = [self.state.data_version]
        with patch.object(resources, "MAX_HISTORY_BYTES", size * 2 + size // 2):
            for number in range(5):
                StateManager.apply_operation("Change", self.state.df.assign(x=lambda frame: frame.x + 1))
                versions.append(self.state.data_version)
            self.assertLessEqual(sum(resources.estimate_bytes(item) for item in self.state.history), resources.MAX_HISTORY_BYTES)
            self.assertGreater(len(self.state.history), 0)
            self.assertLess(len(self.state.history), 5)
            self.assertEqual(len(self.state.operation_log), 5)
            self.assertEqual(self.state.history[-1]["data_version"], versions[-2])
            self.assertTrue(StateManager.undo())
        self.assertEqual(self.state.data_version, versions[-2])
        self.assertEqual(len(self.state.operation_log), 4)

    def test_oversized_checkpoint_is_not_retained(self):
        with patch.object(resources, "MAX_HISTORY_BYTES", 0):
            StateManager.save_snapshot("Too large to retain")
            StateManager.apply_operation("Change", self.state.df.assign(x=[2, 3, 4]))
        self.assertEqual(self.state.history, [])
        self.assertFalse(StateManager.undo())
        self.assertEqual(len(self.state.operation_log), 1)

    def test_chart_count_and_size_rejections_are_atomic(self):
        chart = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        with patch.object(resources, "MAX_CHARTS", 1):
            StateManager.save_chart(chart)
            revision = self.state.project_revision
            with self.assertRaisesRegex(ValueError, "At most"):
                StateManager.save_chart(chart)
            self.assertEqual(len(self.state.viz_charts), 1)
            self.assertEqual(self.state.project_revision, revision)
        chart.data[0].y = [100, 200]
        self.assertEqual(list(self.state.viz_charts[0].data[0].y), [3, 4])
        for field in ("MAX_CHART_BYTES", "MAX_CHART_TOTAL_BYTES"):
            with self.subTest(field=field), patch.object(resources, field, 1):
                with self.assertRaises(ValueError):
                    StateManager.save_chart(chart)
                self.assertEqual(len(self.state.viz_charts), 1)

    def test_experiment_memory_evicts_old_runs_and_rejects_oversized_new_run(self):
        first, second = self.record(extra_bytes=1000), self.record(extra_bytes=1000)
        budget = max(resources.estimate_bytes(first), resources.estimate_bytes(second)) + 500
        with patch.object(resources, "MAX_EXPERIMENT_TOTAL_BYTES", budget):
            StateManager.add_experiment(first)
            StateManager.add_experiment(second)
            self.assertEqual([item["id"] for item in self.state.experiments], [second["id"]])
            previous, revision = copy.deepcopy(self.state.experiments), self.state.project_revision
            with self.assertRaisesRegex(ValueError, "size limit"):
                StateManager.add_experiment(self.record(extra_bytes=budget * 2))
            self.assertEqual([item["id"] for item in self.state.experiments], [item["id"] for item in previous])
            self.assertEqual(self.state.project_revision, revision)
        with patch.object(resources, "MAX_EXPERIMENT_BYTES", 1):
            with self.assertRaises(ValueError):
                StateManager.add_experiment(self.record())

    def test_project_restore_rejects_over_budget_history_or_charts_atomically(self):
        StateManager.apply_operation("Change", self.state.df.assign(x=[2, 3, 4]))
        StateManager.save_chart(go.Figure(go.Bar(x=["a"], y=[1])))
        payload = StateManager.export_project()
        original, revision = self.state.df.copy(), self.state.project_revision
        for field in ("MAX_HISTORY_BYTES", "MAX_CHART_BYTES", "MAX_CHART_TOTAL_BYTES", "MAX_CHARTS"):
            with self.subTest(field=field), patch.object(resources, field, 0):
                with self.assertRaises(ValueError):
                    StateManager.restore_project(payload)
                assert_frame_equal(self.state.df, original)
                self.assertEqual(self.state.project_revision, revision)

    def test_analysis_cache_is_cleared_on_data_change_and_undo(self):
        self.state["_analysis_cache"] = {"cached": "result"}
        self.assertFalse(StateManager.apply_operation("No change", self.state.df))
        self.assertEqual(self.state["_analysis_cache"], {"cached": "result"})
        StateManager.apply_operation("Change", self.state.df.assign(x=[2, 3, 4]))
        self.assertEqual(self.state["_analysis_cache"], {})
        self.state["_analysis_cache"] = {"cached": "other"}
        StateManager.undo()
        self.assertEqual(self.state["_analysis_cache"], {})

    def test_resource_summary_reports_retained_counts_and_sizes(self):
        StateManager.save_snapshot("Checkpoint")
        StateManager.save_chart(go.Figure(go.Bar(x=["a"], y=[1])))
        StateManager.add_experiment(self.record())
        summary = StateManager.resource_summary()
        self.assertEqual(summary["undo_snapshots"], 1)
        self.assertEqual(summary["saved_charts"], 1)
        self.assertEqual(summary["experiments"], 1)
        self.assertEqual(summary["working_data_bytes"], resources.frame_bytes(self.state.df))
        self.assertGreater(summary["undo_bytes"], 0)
        self.assertGreater(summary["saved_chart_json_bytes"], 0)
        self.assertGreater(summary["experiment_bytes"], 0)


class ConcurrentSessionTests(unittest.TestCase):
    def test_concurrent_sessions_keep_frames_history_and_charts_separate(self):
        active = ContextVar("session")
        ready = Barrier(2)

        class SessionProxy:
            def __getattr__(self, key):
                return getattr(active.get(), key)

            def __setattr__(self, key, value):
                setattr(active.get(), key, value)

            def __contains__(self, key):
                return key in active.get()

            def __iter__(self):
                return iter(active.get())

            def __getitem__(self, key):
                return active.get()[key]

            def __setitem__(self, key, value):
                active.get()[key] = value

        def worker(value):
            session = SessionState()
            token = active.set(session)
            try:
                StateManager.init_session()
                StateManager.load_dataset(pd.DataFrame({"x": [value]}), {})
                ready.wait(timeout=10)
                StateManager.apply_operation("Change", session.df.assign(x=[value + 10]))
                StateManager.save_chart(go.Figure(go.Bar(x=["x"], y=[value])))
                ready.wait(timeout=10)
                self.assertEqual(session.df.loc[0, "x"], value + 10)
                self.assertEqual(session.history[0]["df"].loc[0, "x"], value)
                self.assertEqual(list(session.viz_charts[0].data[0].y), [value])
                return session.dataset_id
            finally:
                active.reset(token)

        with patch("backend.state.st.session_state", SessionProxy()), ThreadPoolExecutor(max_workers=2) as pool:
            identifiers = list(pool.map(worker, [1, 100]))
        self.assertNotEqual(*identifiers)


if __name__ == "__main__":
    unittest.main()
