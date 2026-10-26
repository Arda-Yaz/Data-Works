from io import BytesIO
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from backend.experiments import (
    create_experiment, experiment_comparison, experiment_summary,
    export_experiment_csv, export_experiment_json, export_model_artifact,
    importance_figure, predict_artifact, project_experiments,
)
from backend.ingestion import infer_column_meta
from backend.ml import evaluate_model, get_available_models, train_model
from backend.preprocessing import auto_preprocess


def completed_run():
    frame = pd.DataFrame({
        "x": np.arange(40, dtype=float), "kind": ["a", "b"] * 20,
        "target": np.arange(40, dtype=float) * 0.3 + 0.1,
    })
    metadata = infer_column_meta(frame)
    train, test, y_train, y_test, info = auto_preprocess(frame, "target", metadata, problem_type="regression")
    model = train_model(get_available_models("regression")["Linear Regression"], train, y_train)
    metrics = evaluate_model(model, test, y_test, "regression")
    metrics["_model"] = model
    record = create_experiment(
        dataset_id="dataset", data_version="version",
        config={"source": "working", "target": "target", "features": ["x", "kind"],
                "problem_type": "regression", "split_strategy": "random", "test_size": 0.2,
                "group_col": None, "time_col": None, "random_state": 42},
        results={"Linear Regression": metrics}, errors={}, y_test=y_test,
        feature_names=train.columns.tolist(), feature_columns=info["feature_columns"],
        info=info, schema=metadata,
    )
    return record, frame, info


class ExperimentPersistenceTests(unittest.TestCase):
    def test_report_owns_config_metrics_and_targets_without_feature_tables(self):
        record, frame, info = completed_run()
        self.assertEqual(record["dataset_id"], "dataset")
        self.assertEqual(record["data_version"], "version")
        self.assertNotIn("_model", record["results"]["Linear Regression"])
        self.assertNotIn("X_train", record)
        self.assertNotIn("X_test", record)
        self.assertEqual(len(record["y_test"]), 8)
        self.assertEqual(record["feature_columns"], ["x", "kind"])
        self.assertEqual(set(record["models"]), {"Linear Regression"})
        info["scaled"].append("outside")
        self.assertNotIn("outside", record["preprocessing_info"]["scaled"])

    def test_report_only_project_keeps_metrics_and_charts_but_no_live_models(self):
        record, _, _ = completed_run()
        reopened = project_experiments([record])[0]
        self.assertNotIn("models", reopened)
        self.assertNotIn("preprocessor", reopened)
        self.assertEqual(reopened["results"], record["results"])
        self.assertIsNotNone(importance_figure(reopened, "Linear Regression"))
        reopened["config"]["features"].append("changed")
        self.assertNotIn("changed", record["config"]["features"])

    def test_report_exports_are_valid_json_and_comparison_csv(self):
        record, _, _ = completed_run()
        parsed = json.loads(export_experiment_json(record))
        self.assertEqual(parsed["format"], "dataworks-experiment-report")
        self.assertEqual(parsed["experiment"]["id"], record["id"])
        self.assertNotIn("models", parsed["experiment"])
        self.assertNotIn("preprocessor", parsed["experiment"])
        table = pd.read_csv(BytesIO(export_experiment_csv(record)))
        self.assertEqual(table.loc[0, "Model"], "Linear Regression")
        self.assertEqual(table.loc[0, "experiment_id"], record["id"])
        self.assertEqual(experiment_comparison(record).index.name, "Model")

    def test_trusted_model_export_reproduces_predictions_on_reordered_raw_inputs(self):
        record, frame, _ = completed_run()
        # This test loads bytes created above in-process, never external input.
        artifact = joblib.load(BytesIO(export_model_artifact(record, "Linear Regression")))
        source = frame.loc[:4, ["kind", "x"]]
        transformed = record["preprocessor"].transform(source)
        expected = record["models"]["Linear Regression"].predict(
            pd.DataFrame(transformed, index=source.index, columns=record["feature_names"])
        )
        np.testing.assert_allclose(predict_artifact(artifact, source), expected)
        self.assertIn("scikit-learn", artifact["package_versions"])
        self.assertEqual(artifact["feature_columns"], ["x", "kind"])

    def test_reopened_report_requires_retraining_for_model_binary_export(self):
        record, _, _ = completed_run()
        with self.assertRaisesRegex(ValueError, "Retrain"):
            export_model_artifact(experiment_summary(record), "Linear Regression")

    def test_missing_features_and_foreign_artifacts_fail_before_prediction(self):
        record, _, _ = completed_run()
        artifact = joblib.load(BytesIO(export_model_artifact(record, "Linear Regression")))
        with self.assertRaisesRegex(ValueError, "missing"):
            predict_artifact(artifact, pd.DataFrame({"x": [1.0]}))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            predict_artifact({"format": "foreign"}, pd.DataFrame({"x": [1.0]}))

    def test_failed_or_incomplete_run_is_not_retained_as_an_experiment(self):
        with self.assertRaisesRegex(ValueError, "successfully"):
            create_experiment(
                dataset_id="d", data_version="v", config={}, results={}, errors={"failed": "error"},
                y_test=pd.Series([1]), feature_names=["x"], feature_columns=["x"], info={}, schema={},
            )


if __name__ == "__main__":
    unittest.main()
