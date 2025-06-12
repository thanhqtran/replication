# -------------------------------------------------
# --- this script will handle data pre-processing
# == 1. download the necessary data from WDI (put it in the same folder as-is)
# == 2. calculate the average of the 5 years
# == 3. save the data to a csv file
# -------------------------------------------------

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# === input ===
start_year = 1998
end_year = 2002
input_file = 'P_Data_Extract_From_World_Development_Indicators.xlsx'
output_file = 'dataset.csv'


# --- Functions ---
# load data and convert all data to numeric, then calculate average
def load_data(filepath, start_year, end_year):
    df = pd.read_excel(filepath)
    year_cols = df.columns[-(end_year - start_year + 1):]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')
    df['average'] = df[year_cols].mean(axis=1)
    return df
    
# get data series
def get_series(df, name):
    sub_df = df[df['Series Name'] == name]
    sub_df = sub_df[['Country Code', 'average']].copy()
    sub_df.rename(columns={'average': name}, inplace=True)
    return sub_df.reset_index(drop=True)

def merge_data(dfs, names):
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on='Country Code')
    merged.columns = ['country'] + names
    return merged

# --- Workflow ---

# Load and prepare data
df = load_data(input_file, start_year, end_year)
series_list = df['Series Name'].unique()

# Extract needed series
fertility = get_series(df, series_list[0])
infant_mort = get_series(df, series_list[1])
edu = get_series(df, series_list[2])
gni = get_series(df, series_list[3])
pop = get_series(df, series_list[4])

# Merge into one dataset
merged = merge_data(
    [fertility, infant_mort, edu, gni, pop],
    ['fertility', 'infant_mort', 'edu', 'gni', 'pop']
)

# Compute final values
merged = merged.dropna()
merged['net_fertility'] = merged['fertility'] / 2 * (1 - merged['infant_mort'] / 1000)
merged['total_edu'] = merged['edu'] / 100 * merged['gni']
# extract only necessary variables
final = merged[['country', 'net_fertility', 'gni', 'total_edu']]
# rename
final.columns = ['country', 'n', 'y', 'e+theta']
final.to_csv(output_file, index=False)

