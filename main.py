
import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("simulated_call_centre.csv")

# 2. Parse datetime
df["date"] = pd.to_datetime(
    df["date"],
    format="%m/%d/%Y"
)

df["datetime"] = pd.to_datetime(
    df["date"].dt.strftime("%Y-%m-%d")
    + " "
    + df["call_started"]
)

# Ensure service_length is numeric
df["service_length"] = pd.to_numeric(
    df["service_length"],
    errors="coerce"
)

# 3. Create valid service duration
df["valid_service_length"] = (
    df["service_length"].where(
        df["service_length"] > 0,
        np.nan
    )
)

df["invalid_service"] = (
    df["service_length"].isna()
    | (df["service_length"] <= 0)
)

# 4. Aggregate calls and AHT by hour
df["hour_datetime"] = df["datetime"].dt.floor("h")

hourly_agg = (
    df.groupby("hour_datetime")
      .agg(
          call_count=("call_id", "size"),

          aht_seconds=(
              "valid_service_length",
              "mean"
          ),

          valid_service_count=(
              "valid_service_length",
              "count"
          ),

          invalid_service_count=(
              "invalid_service",
              "sum"
          )
      )
)

# 5. Create business-hour timeline
business_dates = pd.bdate_range(
    df["date"].min(),
    df["date"].max()
)

expected_hours = pd.DatetimeIndex([
    day + pd.Timedelta(hours=h)
    for day in business_dates
    for h in range(8, 18)
])

# Reindex without filling AHT with zero
hourly = hourly_agg.reindex(expected_hours)

# No calls means call_count = 0
hourly["call_count"] = (
    hourly["call_count"].fillna(0).astype(int)
)

hourly["valid_service_count"] = (
    hourly["valid_service_count"]
    .fillna(0)
    .astype(int)
)

hourly["invalid_service_count"] = (
    hourly["invalid_service_count"]
    .fillna(0)
    .astype(int)
)

hourly.index.name = "datetime"

# 6. Feature Engineering
hourly["hour"] = hourly.index.hour
hourly["day_of_week"] = hourly.index.dayofweek
hourly["month"] = hourly.index.month

# Long-term trend
hourly["time_index"] = np.arange(len(hourly))

# 7. Lag Features
hourly["lag_1"] = hourly["call_count"].shift(1)
hourly["lag_10"] = hourly["call_count"].shift(10)
hourly["lag_50"] = hourly["call_count"].shift(50)

# 8. Rolling Features
hourly["rolling_mean_10"] = (
    hourly["call_count"]
    .shift(1)
    .rolling(10)
    .mean()
)

hourly["rolling_mean_50"] = (
    hourly["call_count"]
    .shift(1)
    .rolling(50)
    .mean()
)

# 9. Export
hourly.to_csv("hourly_call_dataset(1).csv")


