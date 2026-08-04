import pandas as pd
import numpy as np

# Load dataset

df = pd.read_excel("data/DataSet.xlsm", engine = "openpyxl")
print(df.head())

# drop unnecessary columns

# clean and standardize columns

# normalize categorical columns

# convert journey_date to datetime

# convert time columns : departure_time, arrival_time

# if possible, recover missing values from duration_hours, arrival_time and departure_time columns

# handle remaining missing values

# convert duration_hours to duration_minutes

# extract date and time features

# create route feature

# convert time columns back to HH:MM strings

# save cleaned dataset



