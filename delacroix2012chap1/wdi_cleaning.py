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

# adjust the year here if necessary
start_year = 1998
end_year = 2002

# ========================
# import data
data_raw = pd.read_excel('P_Data_Extract_From_World_Development_Indicators.xlsx')
# create index
df = pd.DataFrame(data_raw, index=range(0, len(data_raw.index)))
# extract all columns names
columns = df.columns
no_cols = end_year - start_year + 1
# extract the last no_cols columns, which contain the data
numeric_cols = df.columns[-no_cols:]
# convert everything to numeric
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # coerce errors to NaN

# ========================
# data preprocessing

# create a new empty column
df['average'] = df[numeric_cols].mean(axis=1)

# A function to extract data from the dataframe
def extract_data(series_name, data_frame):
    series_name = series_name
    data_frame = data_frame
    filtered_df = data_frame[data_frame['Series Name'] == series_name]
    filtered_values = filtered_df[['Country Code', 'average']]
    return filtered_values.reset_index(drop=True)

# extract Series Name
Names = df['Series Name'].unique()

# extract individual data 
fertlity_values = extract_data(Names[0], df)
infant_mortality_values = extract_data(Names[1], df)
edu_values = extract_data(Names[2], df)
gni_values = extract_data(Names[3], df)
pop = extract_data(Names[4], df)

# merge all the 5 dataframes by country code
merged1 = pd.merge(fertlity_values, infant_mortality_values, on='Country Code')
merged2 = pd.merge(merged1, edu_values, on='Country Code')
# rename
merged2.columns = ['Country Code', 'fertility',
                   'infant_mort', 'edu']
merged3 = pd.merge(merged2, gni_values, on='Country Code')
merged4 = pd.merge(merged3, gdp_values, on='Country Code')
# rename
merged4.columns = ['country', 'fertility',
                   'infant_mort', 'edu', 'gni', 'pop']

# ========================
# compute some values

# make a new dataset
data_est = merged4.copy()
# delete all NaN
data_est.dropna(inplace=True)
# calculate net_fertility and total education spending
data_est['net_fertility'] = data_est['fertility'] / 2 * (1  - data_est['infant_mort']/1000)
data_est['total_edu'] = data_est['edu']/100 * data_est['gni']

# =========================
# export the cleaned dataset to csv
data = data_est[['country', 'net_fertility', 'gni', 'total_edu']]
# rename columns
data.columns = ['country', 'n', 'y', 'e+theta']
data.to_csv('dataset.csv')
