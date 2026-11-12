
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# 1. CONFIGURATION
# ==========================================


DATA_PATH = (
    "hourly_call_dataset.csv"
)


SLOTS_PER_WEEK = 50
TEST_WEEKS = 8

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "time_index",
    "lag_1",
    "lag_10",
    "lag_50",
    "rolling_mean_10",
    "rolling_mean_50"
]

TARGET = "call_count"


# ==========================================
# 2. LOAD DATA
# ==========================================

hourly = pd.read_csv(
    DATA_PATH,
    parse_dates=["datetime"],
    index_col="datetime"
)

hourly = hourly.sort_index()

print("Dataset loaded successfully!")
print(f"Total rows: {len(hourly)}")


# Kiểm tra các cột cần thiết
required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in hourly.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

test_size = TEST_WEEKS * SLOTS_PER_WEEK

if len(hourly) <= test_size:
    raise ValueError(
        "Not enough data for the selected test period."
    )

train = hourly.iloc[:-test_size].copy()
test = hourly.iloc[-test_size:].copy()

# Loại bỏ các dòng thiếu feature
train = train.dropna(
    subset=FEATURES + [TARGET]
)

test = test.dropna(
    subset=FEATURES + [TARGET]
)

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")


# ==========================================
# 4. BUILD LIGHTGBM MODEL
# ==========================================

model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Số cuộc gọi không thể âm
y_pred = np.maximum(y_pred, 0)


# ==========================================
# 7. EVALUATION
# ==========================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true, y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    wape = (
        np.abs(y_true - y_pred).sum()
        / y_true.sum()
    ) * 100 if y_true.sum() != 0 else np.nan

    return mae, rmse, wape


# LightGBM metrics
lgbm_mae, lgbm_rmse, lgbm_wape = evaluate(
    y_test, y_pred
)

# Baseline trên chính các dòng test được đánh giá
baseline_pred = test[TARGET].shift(
    SLOTS_PER_WEEK
)

# Lưu ý: baseline cần lấy dữ liệu từ toàn bộ hourly,
# không shift riêng trên test.
baseline_pred = hourly["call_count"].shift(
    SLOTS_PER_WEEK
).reindex(test.index)

valid = baseline_pred.notna()

baseline_mae, baseline_rmse, baseline_wape = evaluate(
    y_test[valid],
    baseline_pred[valid]
)


# ==========================================
# 8. PRINT RESULTS
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print("\nBASELINE")
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"WAPE: {baseline_wape:.2f}%")

print("\nLIGHTGBM")
print(f"MAE:  {lgbm_mae:.4f}")
print(f"RMSE: {lgbm_rmse:.4f}")
print(f"WAPE: {lgbm_wape:.2f}%")

print("======================================")


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

comparison = pd.DataFrame(
    {
        "actual": y_test,
        "baseline_forecast": baseline_pred,
        "lightgbm_forecast": y_pred
    },
    index=test.index
)

comparison["lightgbm_error"] = (
    comparison["actual"]
    - comparison["lightgbm_forecast"]
)

comparison.to_csv(
    "lightgbm_evaluation.csv"
)


# ==========================================
# 10. SAVE METRICS
# ==========================================

metrics = pd.DataFrame({
    "Model": ["Baseline", "LightGBM"],
    "MAE": [baseline_mae, lgbm_mae],
    "RMSE": [baseline_rmse, lgbm_rmse],
    "WAPE": [baseline_wape, lgbm_wape]
})

metrics.to_csv(
    "model_comparison.csv",
    index=False
)


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance:")
print(importance)

print("\nResults saved successfully!")
