
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
).sort_index()

required = FEATURES + [TARGET]

missing = [
    col for col in required
    if col not in hourly.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if hourly.index.has_duplicates:
    raise ValueError("Datetime index contains duplicates.")

print(f"Loaded {len(hourly)} hourly records.")


# ==========================================
# 3. IDENTIFY COMPLETE WORKING WEEKS
# ==========================================

# Xác định thứ Hai của mỗi tuần
week_start = (
    hourly.index.normalize()
    - pd.to_timedelta(hourly.index.dayofweek, unit="D")
)

week_groups = []

for week, group in hourly.groupby(week_start):

    # Chỉ lấy tuần có đủ 50 khung giờ
    if len(group) == SLOTS_PER_WEEK:
        week_groups.append(group)

if len(week_groups) < TEST_WEEKS:
    raise ValueError(
        "Not enough complete weeks for backtesting."
    )

test_groups = week_groups[-TEST_WEEKS:]

print(f"Backtesting {len(test_groups)} weeks.")


# ==========================================
# 4. RECURSIVE FORECAST FUNCTION
# ==========================================

def recursive_forecast(model, history, future_index):

    # Chuyển dữ liệu lịch sử thành danh sách
    values = history[TARGET].tolist()

    predictions = []

    # time_index tiếp tục từ lịch sử
    next_time_index = (
        history["time_index"].iloc[-1] + 1
    )

    for timestamp in future_index:

        if len(values) < SLOTS_PER_WEEK:
            raise ValueError(
                "At least 50 historical slots are required."
            )

        # Tạo feature chỉ từ dữ liệu đã biết
        features = {
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month,
            "time_index": next_time_index,

            "lag_1": values[-1],
            "lag_10": values[-10],
            "lag_50": values[-50],

            "rolling_mean_10": np.mean(values[-10:]),
            "rolling_mean_50": np.mean(values[-50:])
        }

        X = pd.DataFrame(
            [features],
            columns=FEATURES
        )

        prediction = model.predict(X)[0]

        # Số cuộc gọi không thể âm
        prediction = max(0, prediction)

        predictions.append(prediction)

        # Dùng dự báo làm lịch sử cho giờ kế tiếp
        values.append(prediction)

        next_time_index += 1

    return np.array(predictions)


# ==========================================
# 5. METRIC FUNCTION
# ==========================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    wape = (
        np.abs(actual - predicted).sum()
        / actual.sum()
    ) * 100 if actual.sum() != 0 else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }


# ==========================================
# 6. ROLLING-ORIGIN BACKTESTING
# ==========================================

all_results = []
weekly_metrics = []

for week_number, test_week in enumerate(
    test_groups, start=1
):

    # Xác định vị trí bắt đầu tuần kiểm thử
    origin = hourly.index.get_loc(
        test_week.index[0]
    )

    # Chỉ lấy dữ liệu trước tuần kiểm thử
    history = hourly.iloc[:origin].copy()

    # Dữ liệu thực tế của 50 giờ tiếp theo
    actual = test_week[TARGET].to_numpy()

    if len(history) < SLOTS_PER_WEEK:
        continue

    # Chuẩn bị tập huấn luyện
    train = history.dropna(
        subset=FEATURES + [TARGET]
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    # ======================================
    # Train LightGBM
    # ======================================

    model = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # ======================================
    # Forecast 50 hours recursively
    # ======================================

    lgbm_pred = recursive_forecast(
        model,
        history,
        test_week.index
    )

    # ======================================
    # Seasonal Naive Baseline
    # ======================================

    last_week = history[TARGET].iloc[-50:].to_numpy()

    # Lặp lại tuần gần nhất cho tuần dự báo
    baseline_pred = last_week.copy()

    # ======================================
    # Evaluation
    # ======================================

    lgbm_result = calculate_metrics(
        actual, lgbm_pred
    )

    baseline_result = calculate_metrics(
        actual, baseline_pred
    )

    weekly_metrics.append({
        "Week": week_number,
        "Start_Date": test_week.index.min(),
        "End_Date": test_week.index.max(),

        "Baseline_MAE": baseline_result["MAE"],
        "Baseline_RMSE": baseline_result["RMSE"],
        "Baseline_WAPE": baseline_result["WAPE"],

        "LightGBM_MAE": lgbm_result["MAE"],
        "LightGBM_RMSE": lgbm_result["RMSE"],
        "LightGBM_WAPE": lgbm_result["WAPE"]
    })

    # Lưu kết quả từng giờ
    week_result = pd.DataFrame({
        "actual": actual,
        "baseline_forecast": baseline_pred,
        "lightgbm_forecast": lgbm_pred,
        "week_number": week_number
    }, index=test_week.index)

    all_results.append(week_result)

    print(f"\nCompleted week {week_number}/{TEST_WEEKS}")


# ==========================================
# 7. COMBINE RESULTS
# ==========================================

results = pd.concat(all_results)

weekly_df = pd.DataFrame(weekly_metrics)

# Metrics tổng hợp trên toàn bộ 8 tuần
overall_baseline = calculate_metrics(
    results["actual"],
    results["baseline_forecast"]
)

overall_lgbm = calculate_metrics(
    results["actual"],
    results["lightgbm_forecast"]
)

comparison = pd.DataFrame({
    "Model": ["Baseline", "LightGBM"],
    "MAE": [
        overall_baseline["MAE"],
        overall_lgbm["MAE"]
    ],
    "RMSE": [
        overall_baseline["RMSE"],
        overall_lgbm["RMSE"]
    ],
    "WAPE": [
        overall_baseline["WAPE"],
        overall_lgbm["WAPE"]
    ]
})


# ==========================================
# 8. SAVE OUTPUTS
# ==========================================

results.to_csv(
 "weekly_backtest_predictions.csv"
)

weekly_df.to_csv(
 "weekly_backtest_metrics.csv",
    index=False
)

comparison.to_csv(
 "weekly_backtest_comparison.csv",
    index=False
)


# ==========================================
# 9. PRINT SUMMARY
# ==========================================

print("\n========== WEEKLY BACKTEST ==========")

print("\nOverall comparison:")
print(comparison.to_string(index=False))

print("\nMetrics by week:")
print(weekly_df.to_string(index=False))

