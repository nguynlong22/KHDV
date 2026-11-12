
from pathlib import Path
import math
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================


DATA_FILE = (
    "hourly_call_dataset(1).csv"
)

FORECAST_FILE = (
    "weekly_backtest_predictions.csv"
)

TARGET_SERVICE_LEVEL = 0.80
TARGET_WAIT_SECONDS = 20


# ==========================================
# 2. LOAD DATA
# ==========================================

hourly = pd.read_csv(DATA_FILE)
forecast = pd.read_csv(FORECAST_FILE)

hourly["datetime"] = pd.to_datetime(hourly["datetime"])
forecast["datetime"] = pd.to_datetime(forecast["datetime"])

# Dự báo bắt đầu từ thời điểm nào?
forecast_start = forecast["datetime"].min()

# Chỉ sử dụng dữ liệu lịch sử trước thời điểm dự báo
history = hourly[hourly["datetime"] < forecast_start].copy()

history["hour"] = history["datetime"].dt.hour


# ==========================================
# 3. ESTIMATE AHT BY HOUR OF DAY
# ==========================================

history = history[
    (history["aht_seconds"] > 0)
    & (history["valid_service_count"] > 0)
].copy()

# AHT bình quân có trọng số theo số cuộc gọi hợp lệ
history["weighted_aht"] = (
    history["aht_seconds"] * history["valid_service_count"]
)

aht_by_hour = (
    history.groupby("hour")
    .agg(
        total_service_time=("weighted_aht", "sum"),
        total_calls=("valid_service_count", "sum")
    )
)

aht_by_hour["aht_seconds"] = (
    aht_by_hour["total_service_time"]
    / aht_by_hour["total_calls"]
)

global_aht = (
    history["weighted_aht"].sum()
    / history["valid_service_count"].sum()
)


# ==========================================
# 4. ERLANG C FUNCTION
# ==========================================

def calculate_erlang_c(call_rate, aht_seconds, agents):
    """
    call_rate: calls per hour
    aht_seconds: average handling time in seconds
    agents: number of agents
    """

    if call_rate <= 0:
        return 0.0, 1.0

    service_rate = 3600 / aht_seconds

    offered_load = call_rate / service_rate

    utilization = offered_load / agents

    # System must be stable
    if utilization >= 1:
        return 1.0, 0.0

    # Sum A^n / n!, n = 0 ... c-1
    total = 1.0
    term = 1.0

    for n in range(1, agents):
        term *= offered_load / n
        total += term

    # A^c / c!
    last_term = term * offered_load / agents

    # Erlang C denominator
    wait_term = last_term / (1 - utilization)

    p_wait = wait_term / (total + wait_term)

    # Service level within target waiting time
    wait_hours = TARGET_WAIT_SECONDS / 3600

    service_level = (
        1 - p_wait * math.exp(
            -(agents * service_rate - call_rate) * wait_hours
        )
    )

    return p_wait, service_level


# ==========================================
# 5. FIND MINIMUM REQUIRED AGENTS
# ==========================================

def find_required_agents(call_rate, aht_seconds):

    if call_rate <= 0:
        return 0, 0.0, 0.0

    service_rate = 3600 / aht_seconds
    offered_load = call_rate / service_rate

    # Start with a stable system
    agents = max(1, math.floor(offered_load) + 1)

    while agents < 1000:

        p_wait, service_level = calculate_erlang_c(
            call_rate,
            aht_seconds,
            agents
        )

        if service_level >= TARGET_SERVICE_LEVEL:
            return agents, p_wait, service_level

        agents += 1

    raise ValueError("Could not find a feasible staffing level.")


# ==========================================
# 6. CALCULATE STAFFING FOR EACH TIME SLOT
# ==========================================

results = []

for _, row in forecast.iterrows():

    hour = row["datetime"].hour

    call_rate = max(0, float(row["lightgbm_forecast"]))

    # Use historical AHT for the corresponding hour
    if hour in aht_by_hour.index:
        aht = float(aht_by_hour.loc[hour, "aht_seconds"])
    else:
        aht = float(global_aht)

    agents, p_wait, service_level = find_required_agents(
        call_rate,
        aht
    )

    offered_load = call_rate * aht / 3600

    occupancy = (
        offered_load / agents if agents > 0 else 0
    )

    results.append({
        "datetime": row["datetime"],
        "forecast_calls": call_rate,
        "aht_seconds": aht,
        "offered_load_erlang": offered_load,
        "required_agents": agents,
        "probability_wait": p_wait,
        "service_level": service_level,
        "occupancy": occupancy
    })


# ==========================================
# 7. SAVE RESULTS
# ==========================================

staffing = pd.DataFrame(results)

output_file ="staffing_requirements.csv"

staffing.to_csv(output_file, index=False)

print("\nSTAFFING REQUIREMENTS")
print(staffing.head(15).to_string(index=False))

print("\nSUMMARY")
print("Average required agents:",
      round(staffing["required_agents"].mean(), 2))

print("Maximum required agents:",
      staffing["required_agents"].max())

print("Minimum required agents:",
      staffing["required_agents"].min())

print("\nSaved to:", output_file)
