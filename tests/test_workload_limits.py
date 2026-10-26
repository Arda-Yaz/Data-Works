from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

from backend import resources
from backend.ingestion import parse_file
from backend.ml import evaluate_model, train_model
from backend.preprocessing import auto_preprocess


class WorkloadLimitTests(unittest.TestCase):
    def upload(self, contents):
        stream = BytesIO(contents)
        stream.name = "data.csv"
        return stream

    def test_upload_byte_limit_preserves_stream_and_rejects_before_parser(self):
        stream = self.upload(b"x\n123456789\n")
        stream.seek(2)
        with patch.object(resources, "MAX_UPLOAD_BYTES", 5), patch("backend.ingestion.pd.read_csv") as reader:
            with self.assertRaisesRegex(ValueError, "file limit"):
                parse_file(stream)
        self.assertEqual(stream.tell(), 2)
        reader.assert_not_called()

    def test_row_and_cell_limits_reject_parsed_data(self):
        for name, limit in (("MAX_DATASET_ROWS", 2), ("MAX_DATASET_CELLS", 2)):
            with self.subTest(limit=name), patch.object(resources, name, limit):
                with self.assertRaisesRegex(ValueError, "budget"):
                    parse_file(self.upload(b"x,y\n1,a\n2,b\n3,c\n"))

    def test_column_budget_rejects_before_dataframe_allocation(self):
        with patch.object(resources, "MAX_DATASET_COLUMNS", 1), patch("backend.ingestion.pd.read_csv") as reader:
            with self.assertRaisesRegex(ValueError, "column budget"):
                parse_file(self.upload(b"x,y\n1,a\n"))
        reader.assert_not_called()

    def test_upload_summary_is_lazy_and_reused_when_reopened(self):
        from streamlit.testing.v1 import AppTest

        source = '''
import pandas as pd
from backend.state import StateManager
from frontend.pages.data_upload import data_upload
StateManager.init_session()
if not StateManager.has_dataset():
    StateManager.load_dataset(pd.DataFrame({"x": [1.,2.,3.]}), {})
data_upload()
'''
        describe = pd.DataFrame.describe
        with patch.object(pd.DataFrame, "describe", autospec=True, side_effect=describe) as calculation:
            app = AppTest.from_string(source).run()
            self.assertEqual([item.value for item in app.exception], [])
            calculation.assert_not_called()
            app.session_state["ingest_summary_stats"] = True
            app.run()
            self.assertEqual([item.value for item in app.exception], [])
            self.assertEqual(calculation.call_count, 1)
            app.run()
            self.assertEqual(calculation.call_count, 1)

    def test_encoded_memory_budget_checked_before_dense_allocation(self):
        frame = pd.DataFrame({"group": [f"g{i}" for i in range(40)], "target": np.arange(40, dtype=float)})
        with patch.object(resources, "MAX_ML_CELLS", 100), patch("backend.preprocessing._DataFrameColumnTransformer.fit_transform") as fit:
            with self.assertRaisesRegex(ValueError, "memory budget"):
                auto_preprocess(frame, "target", {}, problem_type="regression")
        fit.assert_not_called()

    def test_ml_row_limit_rejects_before_splitting(self):
        frame = pd.DataFrame({"x": range(20), "target": np.arange(20, dtype=float)})
        with patch.object(resources, "MAX_ML_ROWS", 10), patch("backend.preprocessing._split_positions") as split:
            with self.assertRaisesRegex(ValueError, "rows per run"):
                auto_preprocess(frame, "target", {}, problem_type="regression")
        split.assert_not_called()

    def test_kernel_work_limit_rejects_before_fit(self):
        model = SVC()
        with patch.object(resources, "MAX_KERNEL_ROWS", 2), patch.object(model, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "Kernel SVM"):
                train_model(model, pd.DataFrame({"x": [1, 2, 3]}), pd.Series([0, 1, 0]))
        fit.assert_not_called()

    def test_knn_pair_budget_rejects_before_predictions(self):
        features = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        model = train_model(KNeighborsRegressor(), features, pd.Series([1., 2., 3., 4., 5.]))
        with patch.object(resources, "MAX_KNN_PAIRS", 5), patch.object(model, "predict") as predict:
            with self.assertRaisesRegex(ValueError, "pairwise-work budget"):
                evaluate_model(model, features.iloc[:2], pd.Series([1., 2.]), "regression")
        predict.assert_not_called()

    def test_concurrent_training_rejected_then_slot_released(self):
        acquired, release = Event(), Event()

        def active_run():
            with resources.ml_job():
                acquired.set()
                self.assertTrue(release.wait(timeout=5))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(active_run)
            try:
                self.assertTrue(acquired.wait(timeout=5))
                with self.assertRaisesRegex(ValueError, "Another ML run"):
                    with resources.ml_job():
                        self.fail("A concurrent run acquired the slot.")
            finally:
                release.set()
            future.result(timeout=5)
        with resources.ml_job():
            pass

    def test_training_failure_releases_slot(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with resources.ml_job():
                raise RuntimeError("test failure")
        with resources.ml_job():
            pass

    def test_tree_retention_estimate_includes_native_tree_arrays(self):
        from sklearn.ensemble import RandomForestRegressor
        frame = pd.DataFrame({"x": np.arange(100)})
        model = RandomForestRegressor(n_estimators=3, random_state=42).fit(frame, np.arange(100))
        native_bytes = sum(tree.tree_.__getstate__()["nodes"].nbytes for tree in model.estimators_)
        self.assertGreater(resources.estimate_bytes(model), native_bytes)


if __name__ == "__main__":
    unittest.main()
