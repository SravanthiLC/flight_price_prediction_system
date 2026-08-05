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
    df[col] = df[col].astype(str)

    # lowercase
    df[col] = df[col].str.lower()

    # remove # and . and -
    df[col] = df[col].str.replace("#", "", regex = False)
    df[col] = df[col].str.replace(".", "", regex = False)
    df[col] = df[col].str.replace("-", " ", regex = False)

    # remove extra spaces
    df[col] = df[col].str.strip()
    df[col] = df[col].str.replace(r"\s+", " ", regex = True)

# Standardize aircraft types
df["aircraft_type"] = df["aircraft_type"].str.upper()
print(df.head())

# convert journey_date to datetime

# convert time columns : departure_time, arrival_time

# if possible, recover missing values from duration_hours, arrival_time and departure_time columns

# handle remaining missing values

# convert duration_hours to duration_minutes

# extract date and time features

# create route feature

# convert time columns back to HH:MM strings

# save cleaned dataset



