#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 21:12:00 2025

@author: sibo
"""
import pandas as pd

# Read datasets explicitly and skip metadata rows
west_exp = pd.read_csv('/Users/sibo/Desktop/Data/DEUTW/STATS/Exposures_1x1.txt', sep=r"\s+", skiprows=2)
east_exp = pd.read_csv('/Users/sibo/Desktop/Data/DEUTE/STATS/Exposures_1x1.txt', sep=r"\s+", skiprows=2)
total_exp = pd.read_csv('/Users/sibo/Desktop/Data/DEUTNP/STATS/Exposures_1x1.txt', sep=r"\s+", skiprows=2)
west_dt = pd.read_csv('/Users/sibo/Desktop/Data/DEUTW/STATS/Deaths_1x1.txt', sep=r"\s+", skiprows=2)
east_dt = pd.read_csv('/Users/sibo/Desktop/Data/DEUTE/STATS/Deaths_1x1.txt', sep=r"\s+", skiprows=2)
total_dt = pd.read_csv('/Users/sibo/Desktop/Data/DEUTNP/STATS/Deaths_1x1.txt', sep=r"\s+", skiprows=2)

# Explicitly rename columns clearly to avoid confusion
east_exp.columns = ['Year', 'Age', 'Female_East', 'Male_East', 'Total_East']
west_exp.columns = ['Year', 'Age', 'Female_West', 'Male_West', 'Total_West']
total_exp.columns = ['Year', 'Age', 'Female', 'Male', 'Total']

east_dt.columns = ['Year', 'Age', 'Female_East', 'Male_East', 'Total_East']
west_dt.columns = ['Year', 'Age', 'Female_West', 'Male_West', 'Total_West']
total_dt.columns = ['Year', 'Age', 'Female', 'Male', 'Total']

# Convert Age and Year to numeric, Drop rows where Age is NaN, Convert Age and Year to integer
for df in (east_exp, west_exp, total_exp, east_dt, west_dt, total_dt):
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Age']  = pd.to_numeric(df['Age'],  errors='coerce')
    df.dropna(subset=['Year','Age'], inplace=True)
    df["Year"] = df["Year"].astype(int)
    df["Age"]  = df["Age"].astype(int)

# Merge explicitly by Year and Age
ew = pd.merge(
    east_exp,
    west_exp,
    on=['Year','Age'],
    how='inner'
)

dg = pd.merge(
    east_dt,
    west_dt,
    on=['Year','Age'],
    how='inner'
)

# Ages 18–100
keep_cols = ["Year", "Age", "Female", "Male", "Total"]

ew_18 = ew.query(
    "1956 <= Year <= 1989 and 18 <= Age <= 100"
).copy()

ew_18["Female"] = ew_18["Female_East"] + ew_18["Female_West"]
ew_18["Male"]   = ew_18["Male_East"]   + ew_18["Male_West"]
ew_18["Total"]  = ew_18["Total_East"]  + ew_18["Total_West"]

ew_sum_18 = ew_18.loc[:, keep_cols].copy()


dg_18 = dg.query(
    "1956 <= Year <= 1989 and 18 <= Age <= 100"
).copy()

dg_18["Female"] = dg_18["Female_East"] + dg_18["Female_West"]
dg_18["Male"]   = dg_18["Male_East"]   + dg_18["Male_West"]
dg_18["Total"]  = dg_18["Total_East"]  + dg_18["Total_West"]

dg_sum_18 = dg_18.loc[:, keep_cols].copy()


tot_ew_18 = total_exp.query(
    "1990 <= Year <= 2020 and 18 <= Age <= 100"
).loc[:, keep_cols].copy()

tot_dg_18 = total_dt.query(
    "1990 <= Year <= 2020 and 18 <= Age <= 100"
).loc[:, keep_cols].copy()


final_ew_18 = pd.concat(
    [ew_sum_18, tot_ew_18],
    ignore_index=True
).sort_values(["Year", "Age"]).reset_index(drop=True)

final_dg_18 = pd.concat(
    [dg_sum_18, tot_dg_18],
    ignore_index=True
).sort_values(["Year", "Age"]).reset_index(drop=True)

final_ew_18.to_csv(
    "/Users/sibo/Desktop/Data/expo_total_18.csv",
    index=False
)

final_dg_18.to_csv(
    "/Users/sibo/Desktop/Data/deat_total_18.csv",
    index=False
)

# Ages 65–90
ew_65 = ew.query(
    "1956 <= Year <= 1989 and 65 <= Age <= 90"
).copy()

ew_65["Female"] = ew_65["Female_East"] + ew_65["Female_West"]
ew_65["Male"]   = ew_65["Male_East"]   + ew_65["Male_West"]
ew_65["Total"]  = ew_65["Total_East"]  + ew_65["Total_West"]

ew_sum_65 = ew_65.loc[:, keep_cols].copy()


dg_65 = dg.query(
    "1956 <= Year <= 1989 and 65 <= Age <= 90"
).copy()

dg_65["Female"] = dg_65["Female_East"] + dg_65["Female_West"]
dg_65["Male"]   = dg_65["Male_East"]   + dg_65["Male_West"]
dg_65["Total"]  = dg_65["Total_East"]  + dg_65["Total_West"]

dg_sum_65 = dg_65.loc[:, keep_cols].copy()


tot_ew_65 = total_exp.query(
    "1990 <= Year <= 2020 and 65 <= Age <= 90"
).loc[:, keep_cols].copy()

tot_dg_65 = total_dt.query(
    "1990 <= Year <= 2020 and 65 <= Age <= 90"
).loc[:, keep_cols].copy()


final_ew_65 = pd.concat(
    [ew_sum_65, tot_ew_65],
    ignore_index=True
).sort_values(["Year", "Age"]).reset_index(drop=True)

final_dg_65 = pd.concat(
    [dg_sum_65, tot_dg_65],
    ignore_index=True
).sort_values(["Year", "Age"]).reset_index(drop=True)

final_ew_65.to_csv(
    "/Users/sibo/Desktop/Data/expo_total_65.csv",
    index=False
)

final_dg_65.to_csv(
    "/Users/sibo/Desktop/Data/deat_total_65.csv",
    index=False
)