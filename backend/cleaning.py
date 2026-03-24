import streamlit as st
import pandas as pd

def automatic_cleaning(df,missing):
    #automatically perform cleaning steps on missing value based on percentage of the missing data
    
    
    # if missing percentage is less than 5%, drop the rows with missing values for that column
    for col in missing.index:
        if missing.loc[col,"%"] < 5:
            df = df.dropna(subset=[col])
        # if missing percentage is between 5% and 20%, fill the missing values with the median for numerical columns and mode for categorical columns
        elif missing.loc[col,"%"] < 20:
            if df[col].dtype in ["int64","float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        # if missing percentage is greater than 20%, drop the column
        else:
            df = df.drop(columns=[col])
    return df

def find_outliers(df):
    #find outliers in the dataset using the IQR method
    outliers = {}
    for col in df.select_dtypes(include=["int64","float64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
    return outliers

def find_mismatches_by_allowed_values(df, allowed_values_dict):
    
    mismatches = {}

    for col, allowed_values in allowed_values_dict.items():
        col_mismatches = []

        if not allowed_values:  # boşsa skip
            continue

        allowed_set = set(allowed_values)

        for idx, value in df[col].items():
            if pd.isna(value):
                continue

            if str(value) not in allowed_set:
                col_mismatches.append(idx)

        mismatches[col] = col_mismatches

    return mismatches