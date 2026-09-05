import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import bank_churn_vs_code_churn_only as churn


class DummyModel:
    def __init__(self, predictions=None, probabilities=None):
        self.predictions = predictions
        self.probabilities = probabilities
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        return self

    def predict(self, X):
        if self.predictions is not None:
            return np.asarray(self.predictions[: len(X)])
        return np.asarray([0, 1] * ((len(X) + 1) // 2))[: len(X)]

    def predict_proba(self, X):
        if self.probabilities is None:
            probs = np.full(len(X), 0.6)
        else:
            probs = np.asarray(self.probabilities[: len(X)])
        return np.column_stack([1 - probs, probs])


class DummyScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class DummyExplainer:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def shap_values(self, X):
        rows = len(X)
        if self.values.shape[0] == rows:
            return self.values
        return np.tile(self.values[0], (rows, 1))


def make_training_df():
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 2, 3],
            "CustomerId": [101, 102, 102, 103],
            "Surname": ["A", "B", "B", "C"],
            "CreditScore": [650, 700, 700, np.nan],
            "Geography": ["France", "Germany", "Germany", None],
            "Gender": ["Male", "Female", "Female", "Male"],
            "Age": [30, 55, 55, 42],
            "Tenure": [2, 5, 5, 7],
            "Balance": [10000.0, 50000.0, 50000.0, 25000.0],
            "NumOfProducts": [1, 3, 3, 2],
            "HasCrCard": [1, 1, 1, 0],
            "IsActiveMember": [1, 0, 0, 1],
            "EstimatedSalary": [50000.0, 80000.0, 80000.0, 60000.0],
            "Exited": [0, 1, 1, 0],
        }
    )


def test_load_dataset_success(tmp_path, capsys):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({"Age": [30, 40], "Exited": [0, 1]}).to_csv(file_path, index=False)

    result = churn.load_dataset(file_path)

    assert result.shape == (2, 2)
    assert list(result.columns) == ["Age", "Exited"]
    assert "DATASET LOADED SUCCESSFULLY" in capsys.readouterr().out


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        churn.load_dataset(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "column",
    ["Exited", "Churn", "Status", "Target", "Class"],
)
def test_detect_target_supported_names(column):
    assert churn.detect_target(pd.DataFrame({column: [0, 1]})) == column


def test_detect_target_not_found():
    with pytest.raises(ValueError):
        churn.detect_target(pd.DataFrame({"Age": [20, 30]}))


def test_understand_data_runs(capsys):
    df = pd.DataFrame(
        {
            "Age": [20, 30, 30],
            "Exited": [0, 1, 1],
        }
    )

    churn.understand_data(df, "Exited")

    output = capsys.readouterr().out
    assert "DATA UNDERSTANDING" in output
    assert "Target Distribution" in output
    assert "Duplicate Rows" in output


def test_preprocess_training_data_core_behaviour():
    df = make_training_df()

    processed, X, y, encoders = churn.preprocess_training_data(df, "Exited")

    assert len(processed) == 3
    assert "RowNumber" not in processed.columns
    assert "CustomerId" not in processed.columns
    assert "Surname" not in processed.columns
    assert processed.isna().sum().sum() == 0
    assert "Geography" in encoders
    assert "Gender" in encoders
    assert "Balance_Salary_Ratio" in processed.columns
    assert "Age_Group_HighRisk" in processed.columns
    assert "Inactive_Member" in processed.columns
    assert "High_Products" in processed.columns
    assert "Exited" not in X.columns
    assert y.dtype.kind in "iu"


def test_preprocess_training_data_object_target_mapping():
    df = pd.DataFrame(
        {
            "Age": [25, 55],
            "Balance": [1000.0, 2000.0],
            "EstimatedSalary": [5000.0, 6000.0],
            "IsActiveMember": [1, 0],
            "NumOfProducts": [1, 3],
            "Geography": ["France", "Germany"],
            "Exited": ["No", "Yes"],
        }
    )

    processed, _, y, _ = churn.preprocess_training_data(df, "Exited")

    assert processed["Exited"].tolist() == [0, 1]
    assert y.tolist() == [0, 1]


def test_preprocess_training_data_unknown_object_target_uses_encoder():
    df = pd.DataFrame(
        {
            "Age": [25, 55],
            "Geography": ["France", "Germany"],
            "Status": ["Keep", "Leave"],
        }
    )

    processed, _, y, _ = churn.preprocess_training_data(df, "Status")

    assert set(processed["Status"].tolist()) == {0, 1}
    assert set(y.tolist()) == {0, 1}


def test_run_eda_executes_all_plots():
    df = pd.DataFrame(
        {
            "Exited": [0, 1, 0, 1],
            "EstimatedSalary": [1000, 2000, 1500, 3000],
            "CreditScore": [600, 650, 700, 550],
            "NumOfProducts": [1, 2, 1, 3],
            "Age": [30, 45, 35, 55],
        }
    )

    with patch("bank_churn_vs_code_churn_only.plt.show"):
        churn.run_eda(df, "Exited")


def test_split_and_scale_shapes():
    X = pd.DataFrame(
        {
            "A": np.arange(20),
            "B": np.arange(20, 40),
        }
    )
    y = pd.Series([0, 1] * 10)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_train_scaled,
        X_test_scaled,
    ) = churn.split_and_scale(X, y)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4
    assert X_train_scaled.shape == (16, 2)
    assert X_test_scaled.shape == (4, 2)
    assert hasattr(scaler, "mean_")


def test_train_models_without_expensive_training():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    X_test = np.array([[0.0], [1.0]])
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1])

    created = []

    def factory(*args, **kwargs):
        model = DummyModel(predictions=[0, 1])
        created.append(model)
        return model

    with (
        patch("bank_churn_vs_code_churn_only.XGBClassifier", side_effect=factory),
        patch("bank_churn_vs_code_churn_only.LGBMClassifier", side_effect=factory),
        patch("bank_churn_vs_code_churn_only.CatBoostClassifier", side_effect=factory),
    ):
        models, predictions, results = churn.train_models(
            X_train, X_test, y_train, y_test
        )

    assert set(models) == {"XGBoost", "LightGBM", "CatBoost"}
    assert set(predictions) == {"XGBoost", "LightGBM", "CatBoost"}
    assert len(results) == 3
    assert all(created_model.fit_called for created_model in created)
    assert (results["Accuracy"] == 1.0).all()


def test_plot_model_metrics_runs():
    results = pd.DataFrame(
        {
            "Model": ["XGBoost", "LightGBM", "CatBoost"],
            "Accuracy": [0.8, 0.82, 0.84],
            "Precision": [0.7, 0.72, 0.75],
            "Recall": [0.6, 0.65, 0.7],
            "F1-Score": [0.64, 0.68, 0.72],
        }
    )

    with patch("bank_churn_vs_code_churn_only.plt.show") as show:
        churn.plot_model_metrics(results)

    assert show.call_count == 4


def test_evaluate_catboost_runs(capsys):
    model = DummyModel(predictions=[0, 1, 1, 0])
    X_test = np.array([[0], [1], [2], [3]])
    y_test = pd.Series([0, 1, 0, 0])

    with patch("bank_churn_vs_code_churn_only.plt.show"):
        churn.evaluate_catboost(model, X_test, y_test)

    output = capsys.readouterr().out
    assert "CATBOOST PERFORMANCE" in output
    assert "Classification Report" in output


def test_shap_global_explanation_runs():
    X_train = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
        ]
    )
    shap_values = np.array(
        [
            [0.1, -0.2],
            [0.2, 0.1],
            [-0.1, 0.3],
        ]
    )
    fake_explainer = DummyExplainer(shap_values)

    with (
        patch(
            "bank_churn_vs_code_churn_only.shap.TreeExplainer",
            return_value=fake_explainer,
        ),
        patch("bank_churn_vs_code_churn_only.shap.summary_plot"),
    ):
        explainer, importance = churn.shap_global_explanation(
            DummyModel(), X_train, ["Age", "Balance"]
        )

    assert explainer is fake_explainer
    assert list(importance.columns) == ["Feature", "Mean_SHAP"]
    assert len(importance) == 2


def test_shap_global_explanation_handles_list_output():
    class ListExplainer:
        def shap_values(self, X):
            zeros = np.zeros((len(X), 2))
            positives = np.ones((len(X), 2))
            return [zeros, positives]

    with (
        patch(
            "bank_churn_vs_code_churn_only.shap.TreeExplainer",
            return_value=ListExplainer(),
        ),
        patch("bank_churn_vs_code_churn_only.shap.summary_plot"),
    ):
        _, importance = churn.shap_global_explanation(
            DummyModel(), np.array([[1, 2], [3, 4]]), ["A", "B"]
        )

    assert importance["Mean_SHAP"].tolist() == [1.0, 1.0]


def test_save_pipeline_creates_file(tmp_path):
    output_file = tmp_path / "pipeline.pkl"
    scaler = DummyScaler()
    model = DummyModel()

    with patch("bank_churn_vs_code_churn_only.PIPELINE_FILE", output_file):
        churn.save_pipeline(
            model,
            scaler,
            ["Age", "Balance"],
            "Exited",
            {},
        )

    assert output_file.exists()
    saved = churn.joblib.load(output_file)
    assert saved["target_column"] == "Exited"
    assert saved["feature_columns"] == ["Age", "Balance"]


def test_preprocess_new_data_full_path():
    training_df = pd.DataFrame(
        {
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
            "Age": [30, 60],
            "Balance": [10000.0, 20000.0],
            "EstimatedSalary": [50000.0, 60000.0],
            "IsActiveMember": [1, 0],
            "NumOfProducts": [1, 3],
            "Exited": [0, 1],
        }
    )
    _, X, _, encoders = churn.preprocess_training_data(training_df, "Exited")

    pipeline = {
        "target_column": "Exited",
        "feature_columns": X.columns.tolist(),
        "label_encoders": encoders,
    }

    new_data = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [100, 101],
            "Surname": ["A", "B"],
            "Geography": ["Spain", None],
            "Gender": ["Male", "Female"],
            "Age": [55, 35],
            "Balance": [5000.0, np.nan],
            "EstimatedSalary": [40000.0, 45000.0],
            "IsActiveMember": [0, 1],
            "NumOfProducts": [4, 1],
            "Exited": [1, 0],
        }
    )

    result = churn.preprocess_new_data(new_data, pipeline)

    assert list(result.columns) == pipeline["feature_columns"]
    assert result.isna().sum().sum() == 0
    assert "Exited" not in result.columns
    assert result.shape[0] == 2


def test_preprocess_new_data_adds_missing_feature_and_factorizes():
    pipeline = {
        "target_column": "Exited",
        "feature_columns": ["Age", "CityCode", "MissingFeature"],
        "label_encoders": {},
    }
    new_data = pd.DataFrame(
        {
            "Age": [20, 30],
            "CityCode": ["A", "B"],
        }
    )

    result = churn.preprocess_new_data(new_data, pipeline)

    assert list(result.columns) == ["Age", "CityCode", "MissingFeature"]
    assert result["MissingFeature"].tolist() == [0, 0]
    assert result["CityCode"].dtype.kind in "iu"


def test_predict_new_customers_missing_test_file(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.csv"

    with patch("bank_churn_vs_code_churn_only.TEST_FILE", missing):
        churn.predict_new_customers()

    assert "Prediction dataset not found" in capsys.readouterr().out


def test_predict_new_customers_complete_flow(tmp_path):
    test_file = tmp_path / "test.csv"
    pipeline_file = tmp_path / "pipeline.pkl"
    output_file = tmp_path / "predictions.xlsx"

    feature_columns = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    test_data = pd.DataFrame(
        {
            "CustomerId": [1001, 1002],
            "CreditScore": [600, 700],
            "Geography": [0, 1],
            "Gender": [0, 1],
            "Age": [55, 30],
            "Tenure": [3, 7],
            "Balance": [50000.0, 10000.0],
            "NumOfProducts": [3, 1],
            "IsActiveMember": [0, 1],
            "EstimatedSalary": [70000.0, 50000.0],
        }
    )
    test_data.to_csv(test_file, index=False)

    model = DummyModel(probabilities=[0.80, 0.20])
    pipeline = {
        "model": model,
        "scaler": DummyScaler(),
        "feature_columns": feature_columns,
        "target_column": "Exited",
        "label_encoders": {},
    }

    shap_values = np.array(
        [
            [0.8, 0.7, 0.6, 1.0, 0.5, 0.9, 0.85, 0.75, 0.1],
            [-0.2, -0.1, -0.3, -0.4, -0.2, -0.1, -0.2, -0.3, -0.1],
        ]
    )

    with (
        patch("bank_churn_vs_code_churn_only.TEST_FILE", test_file),
        patch("bank_churn_vs_code_churn_only.PIPELINE_FILE", pipeline_file),
        patch("bank_churn_vs_code_churn_only.OUTPUT_FILE", output_file),
        patch("bank_churn_vs_code_churn_only.joblib.load", return_value=pipeline),
        patch(
            "bank_churn_vs_code_churn_only.shap.TreeExplainer",
            return_value=DummyExplainer(shap_values),
        ),
        patch("bank_churn_vs_code_churn_only.plt.show"),
    ):
        churn.predict_new_customers()

    assert output_file.exists()
    result = pd.read_excel(output_file)
    assert len(result) == 1
    assert result.iloc[0]["Customer_ID"] == 1001
    assert result.iloc[0]["Exited_Prediction"] == 1


def test_predict_new_customers_no_customer_id_and_list_shap(tmp_path):
    test_file = tmp_path / "test.csv"
    output_file = tmp_path / "predictions.xlsx"

    feature_columns = ["Age", "Balance"]
    pd.DataFrame(
        {
            "Age": [40, 50],
            "Balance": [1000.0, 2000.0],
        }
    ).to_csv(test_file, index=False)

    pipeline = {
        "model": DummyModel(probabilities=[0.9, 0.1]),
        "scaler": DummyScaler(),
        "feature_columns": feature_columns,
        "target_column": "Exited",
        "label_encoders": {},
    }

    class ListExplainer:
        def shap_values(self, X):
            return [
                np.zeros((len(X), 2)),
                np.array([[0.5, 0.4], [-0.5, -0.4]]),
            ]

    with (
        patch("bank_churn_vs_code_churn_only.TEST_FILE", test_file),
        patch("bank_churn_vs_code_churn_only.OUTPUT_FILE", output_file),
        patch("bank_churn_vs_code_churn_only.joblib.load", return_value=pipeline),
        patch(
            "bank_churn_vs_code_churn_only.shap.TreeExplainer",
            return_value=ListExplainer(),
        ),
        patch("bank_churn_vs_code_churn_only.plt.show"),
    ):
        churn.predict_new_customers()

    result = pd.read_excel(output_file)
    assert result.iloc[0]["Customer_ID"] == 1


def test_main_orchestration():
    df = pd.DataFrame({"Age": [20, 30], "Exited": [0, 1]})
    X = pd.DataFrame({"Age": [20, 30]})
    y = pd.Series([0, 1])
    processed = df.copy()
    split_result = (
        X.iloc[[0]],
        X.iloc[[1]],
        y.iloc[[0]],
        y.iloc[[1]],
        DummyScaler(),
        np.array([[0.0]]),
        np.array([[1.0]]),
    )
    results_df = pd.DataFrame(
        {
            "Model": ["CatBoost"],
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1-Score": [1.0],
        }
    )
    cat_model = DummyModel(predictions=[1])

    with (
        patch("bank_churn_vs_code_churn_only.load_dataset", return_value=df),
        patch("bank_churn_vs_code_churn_only.detect_target", return_value="Exited"),
        patch("bank_churn_vs_code_churn_only.understand_data"),
        patch(
            "bank_churn_vs_code_churn_only.preprocess_training_data",
            return_value=(processed, X, y, {}),
        ),
        patch("bank_churn_vs_code_churn_only.run_eda"),
        patch("bank_churn_vs_code_churn_only.split_and_scale", return_value=split_result),
        patch(
            "bank_churn_vs_code_churn_only.train_models",
            return_value=({"CatBoost": cat_model}, {"CatBoost": np.array([1])}, results_df),
        ),
        patch("bank_churn_vs_code_churn_only.plot_model_metrics"),
        patch("bank_churn_vs_code_churn_only.evaluate_catboost"),
        patch("bank_churn_vs_code_churn_only.shap_global_explanation"),
        patch("bank_churn_vs_code_churn_only.save_pipeline"),
        patch("bank_churn_vs_code_churn_only.predict_new_customers"),
    ):
        churn.main()
