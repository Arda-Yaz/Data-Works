import pandas as pd
import streamlit as st

def _strip_non_numeric(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.replace(r'[^0-9.\-]', '', regex=True)
             .replace('', pd.NA))

def guess_dtype(col: pd.Series) -> pd.Series:
    """Try to coerce an object column to a more appropriate dtype."""
    if col.dtype != 'object':
        return col

    s = col.str.strip()        # might be NaN so convert first if needed

    # numeric?
    num = pd.to_numeric(_strip_non_numeric(s), errors='coerce')
    if num.notna().mean() > 0.8:
        return num

    # date / month‑year / etc.
    dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
    if dt.notna().mean() > 0.8:
        return dt

    # nothing changed
    return col

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with best‑effort conversions applied."""
    return df.apply(guess_dtype)

def dataset_overview(df):
    st.subheader("Dataset Overview")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    missing_values = df.isnull().sum()
    total_missing_values = missing_values.sum()
    columns_with_missing = missing_values[missing_values > 0] 
    if not columns_with_missing.empty:
        st.write(f"Total missing values: {total_missing_values}")
        st.write("Columns with missing values:")
        st.dataframe(columns_with_missing)
    else:
        st.write("No missing values found.")

