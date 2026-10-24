import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from backend.ml import (
    detect_problem_type,
    evaluate_model,
    get_available_models,
    plot_confusion_matrix,
    plot_feature_importance,
    train_model,
)


class ProblemTypeTests(unittest.TestCase):
    def test_fractional_numeric_targets_are_regression_even_with_few_values(self):
        for dtype in ("float64", "Float64", object):
            with self.subTest(dtype=dtype):
                target = pd.Series([0.25, 0.5, 0.25, None], dtype=dtype)
                self.assertEqual(detect_problem_type(target), "regression")

    def test_nullable_boolean_string_and_category_targets_stay_classification(self):
        for target in (
            pd.Series([True, False, None], dtype="boolean"),
            pd.Series(["alpha", "beta", None], dtype="string"),
            pd.Series([0.25, 0.5, None], dtype="category"),
            pd.Series(["1", "2", None], dtype=object),
        ):
            with self.subTest(dtype=target.dtype):
                self.assertEqual(detect_problem_type(target), "classification")

    def test_integral_numeric_suggestion_remains_overridable_heuristic(self):
        self.assertEqual(detect_problem_type(pd.Series([0, 1, 0], dtype="Int64")), "classification")
        self.assertEqual(detect_problem_type(pd.Series(np.arange(30))), "regression")


class ImportanceTests(unittest.TestCase):
    def test_multiclass_coefficients_aggregate_per_feature(self):
        model = SimpleNamespace(coef_=np.array([[1, -2], [-3, 4], [5, -6]]))
        figure = plot_feature_importance(model, ["first", "second"])
        self.assertEqual(list(figure.data[0].y), ["second", "first"])
        np.testing.assert_allclose(figure.data[0].x, [4, 3])
        self.assertIn("Mean absolute coefficient", figure.layout.title.text)

    def test_actual_three_class_logistic_regression_renders_after_training(self):
        features = pd.DataFrame({"x": [0, 0, 1, 1, 2, 2], "y": [0, 1, 0, 1, 1, 2]})
        model = train_model(LogisticRegression(max_iter=1000), features, ["a", "a", "b", "b", "c", "c"])
        figure = plot_feature_importance(model, features.columns.tolist())
        self.assertEqual(set(figure.data[0].y), set(features.columns))
        self.assertEqual(len(figure.data[0].x), 2)

    def test_sparse_linear_svm_coefficients_render(self):
        features = sparse.csr_matrix([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [1, 2]])
        model = train_model(SVC(kernel="linear"), features, [0, 0, 1, 1, 2, 2])
        self.assertTrue(sparse.issparse(model.coef_))
        figure = plot_feature_importance(model, ["first", "second"])
        self.assertEqual(len(figure.data[0].x), 2)

    def test_invalid_importance_dimensions_and_nonfinite_values_are_actionable(self):
        for model, names in (
            (SimpleNamespace(coef_=np.ones((2, 3))), ["only"]),
            (SimpleNamespace(coef_=np.ones((2, 2, 2))), ["a", "b"]),
            (SimpleNamespace(coef_=np.array([[np.nan, 1]])), ["a", "b"]),
            (SimpleNamespace(feature_importances_=np.array([[1, 2]])), ["a", "b"]),
        ):
            with self.subTest(model=model), self.assertRaises(ValueError):
                plot_feature_importance(model, names)

    def test_unsupported_importance_returns_none(self):
        self.assertIsNone(plot_feature_importance(SimpleNamespace(), ["a"]))


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def fixed_model(predictions, classes=None):
        model = Mock()
        model.predict.return_value = np.asarray(predictions)
        model.classes_ = np.asarray(classes) if classes is not None else np.array([])
        return model

    def test_confusion_matrix_preserves_actual_model_class_order(self):
        model = self.fixed_model(["z", "z"], classes=["z", "a"])
        metrics = evaluate_model(model, [[0], [1]], pd.Series(["a", "z"], dtype="string"), "classification")
        self.assertEqual(metrics["class_labels"], ["z", "a"])
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[1, 0], [1, 0]])
        figure = plot_confusion_matrix(metrics["confusion_matrix"], metrics["class_labels"])
        self.assertEqual(list(figure.data[0].x), ["z", "a"])
        self.assertEqual(list(figure.data[0].y), ["z", "a"])

    def test_numeric_confusion_labels_are_displayed_as_categories(self):
        figure = plot_confusion_matrix([[2, 1], [0, 3]], [10, 100])
        self.assertEqual(list(figure.data[0].x), ["10", "100"])
        self.assertEqual(figure.layout.xaxis.type, "category")

    def test_classes_absent_from_training_are_not_silently_omitted(self):
        model = self.fixed_model(["a", "a"], classes=["a", "b"])
        metrics = evaluate_model(model, [[0], [1]], ["a", "new"], "classification")
        self.assertEqual(metrics["class_labels"], ["a", "b", "new"])
        self.assertEqual(int(metrics["confusion_matrix"].sum()), 2)
        self.assertEqual(metrics["confusion_matrix"][2, 0], 1)

    def test_macro_and_per_class_metrics_expose_minority_failure(self):
        model = self.fixed_model(["majority"] * 10, classes=["majority", "minority"])
        metrics = evaluate_model(model, np.arange(10).reshape(-1, 1), ["majority"] * 9 + ["minority"], "classification")
        self.assertLess(metrics["macro_f1"], metrics["f1"])
        minority = next(item for item in metrics["per_class"] if item["class"] == "minority")
        self.assertEqual(minority["support"], 1)
        self.assertEqual(minority["recall"], 0)

    def test_undefined_r_squared_is_unavailable_without_hiding_other_metrics(self):
        for actual, predictions in (([1.0], [1.5]), ([2.0, 2.0], [2.0, 2.0])):
            with self.subTest(actual=actual):
                model = self.fixed_model(predictions)
                metrics = evaluate_model(model, np.zeros((len(actual), 1)), actual, "regression")
                self.assertIsNone(metrics["R²"])
                self.assertTrue(metrics["r2_unavailable_reason"])
                self.assertIsInstance(metrics["MAE"], float)

    def test_regular_regression_keeps_finite_r_squared(self):
        model = self.fixed_model([1.0, 2.0, 3.0])
        metrics = evaluate_model(model, [[0], [1], [2]], [1.0, 2.0, 3.0], "regression")
        self.assertEqual(metrics["R²"], 1)
        self.assertIsNone(metrics["r2_unavailable_reason"])

    def test_finite_targets_cannot_silently_produce_infinite_error_metrics(self):
        model = self.fixed_model([-1e308, 1e308])
        with self.assertRaisesRegex(ValueError, "rescale"):
            evaluate_model(model, [[0], [1]], [1e308, -1e308], "regression")

    def test_invalid_confusion_matrix_shapes_are_rejected(self):
        for matrix, labels in (([[1, 2, 3]], None), ([[1, 0], [0, 1]], ["one"]), ([], None)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                plot_confusion_matrix(matrix, labels)

    def test_unknown_problem_type_is_rejected_before_prediction(self):
        model = self.fixed_model([0, 1])
        with self.assertRaises(ValueError):
            evaluate_model(model, [[0], [1]], [0, 1], "unknown")
        model.predict.assert_not_called()
        with self.assertRaises(ValueError):
            get_available_models("unknown")

    def test_empty_missing_and_nonfinite_evaluation_targets_are_rejected(self):
        for target in ([], [np.nan], [np.inf], [pd.NA]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                evaluate_model(self.fixed_model([0]), np.zeros((len(target), 1)), target, "regression")


class TrainingValidationTests(unittest.TestCase):
    def test_classification_requires_two_discrete_training_classes(self):
        for target in (["one"] * 3, [0.1, 0.2, 0.3]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                train_model(LogisticRegression(), [[0], [1], [2]], target)

    def test_invalid_training_targets_are_rejected_before_fitting(self):
        for target in ([], [0, np.nan], [0, np.inf], [1 + 2j, 3 + 4j]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                train_model(LinearRegression(), np.zeros((len(target), 1)), target)

    def test_regression_does_not_convert_dates_to_epoch_numbers(self):
        with self.assertRaisesRegex(ValueError, "dates"):
            train_model(LinearRegression(), [[0], [1]], pd.to_datetime(["2024-01-01", "2024-01-02"]))

    def test_feature_and_target_row_counts_must_agree(self):
        with self.assertRaises(ValueError):
            train_model(LinearRegression(), [[0], [1]], [0])

    def test_knn_reports_too_few_training_rows_before_prediction(self):
        with self.assertRaisesRegex(ValueError, "5 training rows"):
            train_model(KNeighborsClassifier(), [[0], [1], [2], [3]], [0, 1, 0, 1])

    def test_literal_nan_class_label_remains_a_label(self):
        model = train_model(LogisticRegression(), [[0], [1], [2], [3]], ["NaN", "valid", "NaN", "valid"])
        self.assertEqual(set(model.classes_), {"NaN", "valid"})


if __name__ == "__main__":
    unittest.main()
