
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# 1. CONFIGURATION
# ==========================================


DATA_PATH = "hourly_call_dataset.csv"


SLOTS_PER_WEEK = 50
TEST_WEEKS = 8


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
print(f"Number of rows: {len(hourly)}")
print(hourly.head())


# ==========================================
# 3. BASELINE FORECAST
# ==========================================

# Dự báo bằng số cuộc gọi cùng khung giờ tuần trước
hourly["baseline_forecast"] = (
    hourly["call_count"].shift(SLOTS_PER_WEEK)
)


# ==========================================
# 4. SELECT TEST DATA
# ==========================================

test_size = TEST_WEEKS * SLOTS_PER_WEEK

test = hourly.tail(test_size).copy()

test = test.dropna(
    subset=["baseline_forecast"]
)

y_true = test["call_count"]
y_pred = test["baseline_forecast"]


# ==========================================
# 5. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

wape = (
    np.abs(y_true - y_pred).sum()
    / y_true.sum()
) * 100 if y_true.sum() != 0 else np.nan


print("\n========== BASELINE EVALUATION ==========")
print(f"MAE:  {mae:.2f} calls")
print(f"RMSE: {rmse:.2f} calls")
print(f"WAPE: {wape:.2f}%")
print("=========================================")


# ==========================================
# 6. SAVE PREDICTIONS
# ==========================================

comparison = test[
    ["call_count", "baseline_forecast"]
].copy()

comparison["error"] = (
    comparison["call_count"]
    - comparison["baseline_forecast"]
)

comparison.to_csv(
    "baseline_evaluation.csv"
)


# ==========================================
# 7. SAVE METRICS
# ==========================================

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "WAPE"],
    "Value": [mae, rmse, wape]
})

metrics.to_csv(
    "baseline_metrics.csv",
    index=False
)


print("\nResults saved successfully!")
print(f"Predictions: {'baseline_evaluation.csv'}")
print(f"Metrics: {'baseline_metrics.csv'}")

print("\nSample predictions:")
print(comparison.head(20))
