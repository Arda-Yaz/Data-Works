import pandas as pd
import streamlit as st

from backend.ingestion import apply_inferred_dtypes, infer_column_meta, infer_dtype

def guess_dtype(col: pd.Series) -> pd.Series:
    """Use the same conservative conversion rules as the schema page."""
    target, _ = infer_dtype(col)
    frame = col.to_frame(name="value")
    result = apply_inferred_dtypes(frame, {"value": {"inferred_dtype": target}})
    return result["value"].rename(col.name)

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy converted using the shared schema rules."""
    return apply_inferred_dtypes(df, infer_column_meta(df))

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

