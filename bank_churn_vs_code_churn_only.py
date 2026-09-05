# Bank Customer Churn Prediction - VS Code Version
# Models: XGBoost, LightGBM, CatBoost
# Software Metrics-ready structure

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "Churn.csv"
TEST_FILE = BASE_DIR / "Churn_Testing_500.csv"
PIPELINE_FILE = BASE_DIR / "bank_churn_pipeline.pkl"
OUTPUT_FILE = BASE_DIR / "churn_predictions.xlsx"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place Churn.csv in the same folder as this Python file."
        )
    df = pd.read_csv(path)
    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    return df


def detect_target(df: pd.DataFrame) -> str:
    possible_targets = ["Exited", "Churn", "Status", "Target", "Class"]
    for col in possible_targets:
        if col in df.columns:
            return col
    raise ValueError("Target column not found. Expected one of: " + ", ".join(possible_targets))


def understand_data(df: pd.DataFrame, target_col: str) -> None:
    print("\n" + "=" * 60)
    print("DATA UNDERSTANDING")
    print("=" * 60)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nLast 5 Rows:\n", df.tail())
    print("\nData Types:\n", df.dtypes)
    print("\nStatistical Summary:\n", df.describe(include="all"))
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nTarget Distribution:\n", df[target_col].value_counts())
    print(
        "\nTarget Percentage:\n",
        (df[target_col].value_counts(normalize=True) * 100).round(2),
    )


def preprocess_training_data(df: pd.DataFrame, target_col: str):
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])

    # Remove unnecessary columns
    remove_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    if remove_cols:
        df = df.drop(columns=remove_cols)

    # Convert target to numeric if needed
    if df[target_col].dtype == "object":
        target_mapping = {
            "Yes": 1,
            "No": 0,
            "Exited": 1,
            "Stayed": 0,
            "Churn": 1,
            "Not Churn": 0,
        }
        df[target_col] = df[target_col].replace(target_mapping)
        if df[target_col].dtype == "object":
            target_encoder = LabelEncoder()
            df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

    # Encode feature categoricals
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Feature engineering retained from uploaded Colab code
    if "Balance" in df.columns and "EstimatedSalary" in df.columns:
        df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    if "Age" in df.columns:
        df["Age_Group_HighRisk"] = (df["Age"] > 50).astype(int)

    if "IsActiveMember" in df.columns:
        df["Inactive_Member"] = (df["IsActiveMember"] == 0).astype(int)

    if "NumOfProducts" in df.columns:
        df["High_Products"] = (df["NumOfProducts"] >= 3).astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return df, X, y, label_encoders


def run_eda(df: pd.DataFrame, target_col: str) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    sns.set_theme(style="whitegrid")

    target_counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(target_counts.index.astype(str), target_counts.values)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
        )
    plt.title("Target Variable Distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    if "EstimatedSalary" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=target_col, y="EstimatedSalary", data=df)
        plt.title("Estimated Salary vs Churn")
        plt.tight_layout()
        plt.show()

    if "CreditScore" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=target_col, y="CreditScore", data=df)
        plt.title("Credit Score vs Churn")
        plt.tight_layout()
        plt.show()

    if "NumOfProducts" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=target_col, y="NumOfProducts", data=df)
        plt.title("Number of Products vs Churn")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape :", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled


def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }

    predictions = {}
    results = []

    print("\n" + "=" * 60)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        predictions[name] = pred

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
        }
        results.append(result)

        print(
            f"{name:10s} | "
            f"Accuracy={result['Accuracy']:.4f} | "
            f"Precision={result['Precision']:.4f} | "
            f"Recall={result['Recall']:.4f} | "
            f"F1={result['F1-Score']:.4f}"
        )

    results_df = pd.DataFrame(results)
    return models, predictions, results_df


def plot_model_metrics(results_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        bars = plt.bar(results_df["Model"], results_df[metric])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        plt.title(f"{metric} Comparison of Machine Learning Models")
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


def evaluate_catboost(cat_model, X_test_scaled, y_test):
    cat_pred = cat_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, cat_pred)
    precision = precision_score(y_test, cat_pred, zero_division=0)
    recall = recall_score(y_test, cat_pred, zero_division=0)
    f1 = f1_score(y_test, cat_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("CATBOOST PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, cat_pred))

    cm = confusion_matrix(y_test, cat_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("CatBoost Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def shap_global_explanation(cat_model, X_train_scaled, feature_names):
    print("\n" + "=" * 60)
    print("SHAP GLOBAL EXPLANATION")
    print("=" * 60)

    X_shap = pd.DataFrame(X_train_scaled, columns=feature_names)

    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Mean_SHAP": mean_shap}
    ).sort_values("Mean_SHAP", ascending=False)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10))

    shap.summary_plot(shap_values, X_shap, show=True)
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=True)

    return explainer, importance_df


def save_pipeline(cat_model, scaler, feature_columns, target_col, label_encoders):
    pipeline = {
        "model": cat_model,
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "target_column": target_col,
        "label_encoders": label_encoders,
    }

    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"\nPipeline saved successfully: {PIPELINE_FILE}")


def preprocess_new_data(new_data, pipeline):
    new_data = new_data.copy()

    target_col = pipeline["target_column"]
    feature_columns = pipeline["feature_columns"]
    label_encoders = pipeline.get("label_encoders", {})

    if target_col in new_data.columns:
        new_data = new_data.drop(columns=[target_col])

    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in new_data.columns:
            new_data = new_data.drop(columns=[col])

    numeric_cols = new_data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        new_data[col] = new_data[col].fillna(new_data[col].median())

    categorical_cols = new_data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        mode = new_data[col].mode()
        if not mode.empty:
            new_data[col] = new_data[col].fillna(mode[0])

    for col, encoder in label_encoders.items():
        if col in new_data.columns:
            values = new_data[col].astype(str)
            known_classes = set(encoder.classes_)

            fallback = encoder.classes_[0]
            values = values.apply(lambda x: x if x in known_classes else fallback)
            new_data[col] = encoder.transform(values)

    # Re-create the same engineered features used during training
    if "Balance" in new_data.columns and "EstimatedSalary" in new_data.columns:
        new_data["Balance_Salary_Ratio"] = (
            new_data["Balance"] / (new_data["EstimatedSalary"] + 1)
        )

    if "Age" in new_data.columns:
        new_data["Age_Group_HighRisk"] = (new_data["Age"] > 50).astype(int)

    if "IsActiveMember" in new_data.columns:
        new_data["Inactive_Member"] = (new_data["IsActiveMember"] == 0).astype(int)

    if "NumOfProducts" in new_data.columns:
        new_data["High_Products"] = (new_data["NumOfProducts"] >= 3).astype(int)

    # Convert any remaining object columns
    for col in new_data.select_dtypes(include=["object"]).columns:
        new_data[col] = pd.factorize(new_data[col])[0]

    for col in feature_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[feature_columns]
    return new_data


def predict_new_customers():
    if not TEST_FILE.exists():
        print(
            "\nPrediction dataset not found, so deployment prediction is skipped.\n"
            f"To enable it, place {TEST_FILE.name} in the project folder."
        )
        return

    pipeline = joblib.load(PIPELINE_FILE)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    feature_columns = pipeline["feature_columns"]

    new_data = pd.read_csv(TEST_FILE)
    original_data = new_data.copy()

    processed = preprocess_new_data(new_data, pipeline)
    new_scaled = scaler.transform(processed)

    probability = model.predict_proba(new_scaled)[:, 1]
    prediction = (probability > 0.5).astype(int)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(new_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Mean_SHAP": mean_shap}
    ).sort_values("Mean_SHAP", ascending=False)

    plt.figure(figsize=(9, 6))
    plt.barh(importance_df["Feature"], importance_df["Mean_SHAP"])
    plt.gca().invert_yaxis()
    plt.title("Global Feature Importance")
    plt.xlabel("Mean |SHAP|")
    plt.tight_layout()
    plt.show()

    reasons_list = []

    for row in shap_values:
        top_index = np.argsort(np.abs(row))[::-1][:5]
        reasons = []

        for idx in top_index:
            feature = feature_columns[idx]
            impact = row[idx]

            if impact <= 0:
                continue

            if feature == "Age":
                reasons.append("Higher age")
            elif feature == "Balance":
                reasons.append("Balance impact")
            elif feature == "IsActiveMember":
                reasons.append("Inactive membership")
            elif feature == "NumOfProducts":
                reasons.append("Number of products impact")
            elif feature == "CreditScore":
                reasons.append("Credit score impact")
            elif feature == "Geography":
                reasons.append("Geography impact")
            elif feature == "Gender":
                reasons.append("Gender impact")
            elif feature == "Tenure":
                reasons.append("Tenure impact")

        reasons = list(dict.fromkeys(reasons))
        if reasons:
            reasons_list.append("; ".join(reasons[:3]))
        else:
            reasons_list.append("No strong churn reason identified")

    customer_col = None
    for col in ["CustomerId", "customer_id", "Customer_ID", "customerid"]:
        if col in original_data.columns:
            customer_col = col
            break

    if customer_col:
        customer_ids = original_data[customer_col]
    else:
        customer_ids = np.arange(1, len(original_data) + 1)

    result = pd.DataFrame(
        {
            "Customer_ID": customer_ids,
            "Exited_Prediction": prediction,
            "Churn_Probability": probability.round(3),
            "Explanation": reasons_list,
        }
    )

    # Keep only customers predicted to churn (Exited_Prediction = 1)
    churn_result = result[result["Exited_Prediction"] == 1].copy()

    print("\nCustomers Predicted to Churn:")
    print(churn_result.head(10))
    print(f"\nTotal customers predicted to churn: {len(churn_result)}")

    # Save only churn customers to Excel
    churn_result.to_excel(OUTPUT_FILE, index=False)
    print(f"\nExcel file saved with churn customers only: {OUTPUT_FILE}")


def main():
    print("=" * 60)
    print("BANK CUSTOMER CHURN PREDICTION")
    print("XGBoost + LightGBM + CatBoost")
    print("=" * 60)

    df = load_dataset(TRAIN_FILE)
    target_col = detect_target(df)

    understand_data(df, target_col)

    processed_df, X, y, label_encoders = preprocess_training_data(df, target_col)

    run_eda(processed_df, target_col)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_train_scaled,
        X_test_scaled,
    ) = split_and_scale(X, y)

    models, predictions, results_df = train_models(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    print("\nModel Comparison:")
    print(results_df.round(4))

    plot_model_metrics(results_df)

    cat_model = models["CatBoost"]
    evaluate_catboost(cat_model, X_test_scaled, y_test)

    shap_global_explanation(
        cat_model,
        X_train_scaled,
        X.columns.tolist(),
    )

    save_pipeline(
        cat_model,
        scaler,
        X.columns.tolist(),
        target_col,
        label_encoders,
    )

    predict_new_customers()

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
