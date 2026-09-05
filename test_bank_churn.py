import pandas as pd
import numpy as np
import pytest

from bank_churn_vs_code_churn_only import (
    detect_target,
    preprocess_training_data,
    split_and_scale,
    preprocess_new_data,
)


# ============================================================
# TEST 1: TARGET COLUMN DETECTION
# ============================================================

def test_detect_target():

    df = pd.DataFrame({
        "Age": [30, 40, 50],
        "Exited": [0, 1, 0]
    })

    target = detect_target(df)

    assert target == "Exited"


# ============================================================
# TEST 2: TARGET COLUMN NOT FOUND
# ============================================================

def test_detect_target_not_found():

    df = pd.DataFrame({
        "Age": [30, 40],
        "Balance": [1000, 2000]
    })

    with pytest.raises(ValueError):
        detect_target(df)


# ============================================================
# TEST 3: REMOVE DUPLICATES
# ============================================================

def test_preprocessing_removes_duplicates():

    df = pd.DataFrame({
        "CreditScore": [600, 600, 700],
        "Geography": ["France", "France", "Spain"],
        "Gender": ["Male", "Male", "Female"],
        "Age": [40, 40, 35],
        "Tenure": [5, 5, 3],
        "Balance": [50000, 50000, 70000],
        "NumOfProducts": [2, 2, 1],
        "HasCrCard": [1, 1, 1],
        "IsActiveMember": [1, 1, 0],
        "EstimatedSalary": [60000, 60000, 80000],
        "Exited": [0, 0, 1]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    # Two identical rows should become one
    assert len(processed_df) == 2


# ============================================================
# TEST 4: CATEGORICAL VALUES ARE ENCODED
# ============================================================

def test_categorical_encoding():

    df = pd.DataFrame({
        "CreditScore": [600, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"],
        "Age": [40, 35],
        "Tenure": [5, 3],
        "Balance": [50000, 70000],
        "NumOfProducts": [2, 1],
        "HasCrCard": [1, 1],
        "IsActiveMember": [1, 0],
        "EstimatedSalary": [60000, 80000],
        "Exited": [0, 1]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    assert pd.api.types.is_numeric_dtype(X["Geography"])
    assert pd.api.types.is_numeric_dtype(X["Gender"])

    assert "Geography" in encoders
    assert "Gender" in encoders


# ============================================================
# TEST 5: FEATURE ENGINEERING
# ============================================================

def test_feature_engineering():

    df = pd.DataFrame({
        "CreditScore": [600, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"],
        "Age": [55, 35],
        "Tenure": [5, 3],
        "Balance": [50000, 70000],
        "NumOfProducts": [3, 1],
        "HasCrCard": [1, 1],
        "IsActiveMember": [0, 1],
        "EstimatedSalary": [60000, 80000],
        "Exited": [1, 0]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    assert "Balance_Salary_Ratio" in X.columns
    assert "Age_Group_HighRisk" in X.columns
    assert "Inactive_Member" in X.columns
    assert "High_Products" in X.columns


# ============================================================
# TEST 6: X AND y CREATION
# ============================================================

def test_features_and_target():

    df = pd.DataFrame({
        "CreditScore": [600, 700, 650, 720],
        "Geography": ["France", "Germany", "Spain", "France"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Age": [40, 35, 45, 50],
        "Tenure": [5, 3, 6, 4],
        "Balance": [50000, 70000, 60000, 80000],
        "NumOfProducts": [2, 1, 2, 1],
        "HasCrCard": [1, 1, 0, 1],
        "IsActiveMember": [1, 0, 1, 0],
        "EstimatedSalary": [60000, 80000, 75000, 90000],
        "Exited": [0, 1, 0, 1]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    assert "Exited" not in X.columns

    assert len(X) == len(y)

    assert list(y) == [0, 1, 0, 1]


# ============================================================
# TEST 7: TRAIN-TEST SPLIT AND SCALING
# ============================================================

def test_split_and_scale():

    X = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Balance": [
            1000, 2000, 3000, 4000, 5000,
            6000, 7000, 8000, 9000, 10000
        ]
    })

    y = pd.Series([
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1
    ])

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_train_scaled,
        X_test_scaled
    ) = split_and_scale(X, y)

    assert len(X_train) == 8
    assert len(X_test) == 2

    assert X_train_scaled.shape == (8, 2)
    assert X_test_scaled.shape == (2, 2)

    assert scaler is not None


# ============================================================
# TEST 8: MISSING VALUE HANDLING
# ============================================================

def test_missing_value_handling():

    df = pd.DataFrame({
        "CreditScore": [600, np.nan, 700],
        "Geography": ["France", None, "Germany"],
        "Gender": ["Male", "Female", None],
        "Age": [40, 35, np.nan],
        "Tenure": [5, 3, 2],
        "Balance": [50000, np.nan, 70000],
        "NumOfProducts": [2, 1, 2],
        "HasCrCard": [1, 1, 0],
        "IsActiveMember": [1, 0, 1],
        "EstimatedSalary": [60000, 80000, 70000],
        "Exited": [0, 1, 0]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    assert processed_df.isnull().sum().sum() == 0


# ============================================================
# TEST 9: UNNECESSARY COLUMNS ARE REMOVED
# ============================================================

def test_remove_unnecessary_columns():

    df = pd.DataFrame({
        "RowNumber": [1, 2],
        "CustomerId": [1001, 1002],
        "Surname": ["A", "B"],
        "CreditScore": [600, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"],
        "Age": [40, 35],
        "Tenure": [5, 3],
        "Balance": [50000, 70000],
        "NumOfProducts": [2, 1],
        "HasCrCard": [1, 1],
        "IsActiveMember": [1, 0],
        "EstimatedSalary": [60000, 80000],
        "Exited": [0, 1]
    })

    processed_df, X, y, encoders = preprocess_training_data(
        df,
        "Exited"
    )

    assert "RowNumber" not in X.columns
    assert "CustomerId" not in X.columns
    assert "Surname" not in X.columns