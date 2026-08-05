import pandas as pd
import numpy as np

# Load dataset

df = pd.read_excel("data/DataSet.xlsm", engine = "openpyxl")

# drop unnecessary columns

df.drop(columns=["flight_id"], inplace = True)

# clean and standardize columns
text_columns = [
    "airline",
    "source_city",
    "destination_city",
    "travel_class",
    "season",
    "day_of_week",
    "aircraft_type"
]

for col in text_columns:
    # convert to string
    df[col] = df[col].astype("string")

    df[col] = (
    df[col]
    .str.lower()
    .str.replace("#", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace("-", " ", regex=False)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    )

# Standardize aircraft types
df["aircraft_type"] = df["aircraft_type"].str.upper()
print(df.head())

# convert journey_date to datetime

df["journey_date"] = pd.to_datetime(
    df["journey_date"],
    errors = "coerce"
)

# convert time columns departure_time, arrival_time into pandas datetime format

df["departure_time"] = pd.to_datetime(
    df["departure_time"].astype(str),
    format="%H:%M:%S",
    errors="coerce"
)

df["arrival_time"] = pd.to_datetime(
    df["arrival_time"].astype(str),
    format="%H:%M:%S",
    errors="coerce"
)

# if possible, recover missing values from duration_hours, arrival_time and departure_time columns

duration_minutes = (df["duration_hours"] * 60).round()

# recover missing arrival time
mask = (
    df["arrival_time"].isna()
    & df["departure_time"].notna()
    & duration_minutes.notna()
)

df.loc[mask, "arrival_time"] = (
    df.loc[mask, "departure_time"] +
    pd.to_timedelta(duration_minutes[mask], unit="m")
)

# Recover missing departure_time
mask = (
    df["departure_time"].isna()
    & df["arrival_time"].notna()
    & duration_minutes.notna()
)

df.loc[mask, "departure_time"] = (
    df.loc[mask, "arrival_time"] -
    pd.to_timedelta(duration_minutes[mask], unit="m")
)

# Calculate missing duration_minutes from times
mask = (
    duration_minutes.isna()
    & df["departure_time"].notna()
    & df["arrival_time"].notna()
)

time_diff = (
    df.loc[mask, "arrival_time"] -
    df.loc[mask, "departure_time"]
).dt.total_seconds() / 60

# Handle overnight flights
time_diff = np.where(time_diff < 0, time_diff + 24 * 60, time_diff)

duration_minutes.loc[mask] = time_diff

# Store final duration_minutes
df["duration_minutes"] = duration_minutes

# Remove duration_hours since duration_minutes is the final feature
df.drop(columns=["duration_hours"], inplace=True)

# handle remaining missing values

# Display missing percentage
missing_percent = df.isnull().mean() * 100
print("Missing values before imputation (%):")
print(missing_percent)

# Fill numerical columns with median
numeric_cols = [
    "booking_window_days",
    "distance_km",
    "rating",
    "duration_minutes"
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = [
    "airline",
    "source_city",
    "destination_city",
    "travel_class",
    "season",
    "day_of_week",
    "aircraft_type"
]

for col in categorical_cols:
    mode_value = df[col].mode(dropna=True)
    if not mode_value.empty:
        df[col] = df[col].fillna(mode_value[0])

# Drop rows where date or time is still missing
print("Rows before dropping:", len(df))
print(df[["journey_date", "departure_time", "arrival_time"]].isna().sum())
df.dropna(
    subset=["journey_date", "departure_time", "arrival_time"],
    inplace=True
)

df["duration_minutes"] = df["duration_minutes"].round().astype(int)

# extract date and time features

# Date features
df["journey_month"] = df["journey_date"].dt.month
df["journey_day"] = df["journey_date"].dt.day

# Time features
df["departure_hour"] = df["departure_time"].dt.hour
df["arrival_hour"] = df["arrival_time"].dt.hour

df["departure_time"] = df["departure_time"].dt.strftime("%H:%M")
df["arrival_time"] = df["arrival_time"].dt.strftime("%H:%M")

# create route feature

df["route"] = df["source_city"] + " -> " + df["destination_city"]

# save cleaned dataset

df.to_csv("data/cleaned_dataset.csv", index=False)

print("\nPreprocessing complete.")
print(df.head())




