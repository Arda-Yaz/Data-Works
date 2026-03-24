import streamlit as st
from backend.cleaning import automatic_cleaning, find_mismatches_by_allowed_values, find_outliers

def data_cleaning():
    st.title("Data Cleaning")
    st.write("This page will allow you to clean your dataset. You can choose to accept the guessed types, or edit them manually.")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset in the Preview page first.")
        return
    
    df = st.session_state.df


    st.dataframe(df.head(5))

    with st.expander("Missing values"):
        missing = df.isna().sum().to_frame("missing")
        missing["%"] = 100 * missing["missing"] / len(df)
        st.dataframe(missing)

    # Collect expected types for each column
    st.subheader("Specify Allowed Values (Optional)")

    allowed_values_dict = {}

    for col in df.columns:
        allowed_input = st.text_input(
            f"Allowed values for '{col}' (comma separated):",
            key=f"allowed_{col}"
        )

        if allowed_input:
            allowed_values = [v.strip() for v in allowed_input.split(",")]
            allowed_values_dict[col] = allowed_values

    #button to find potential mismatches
    if st.button("Find mismatches (allowed values based)"):
        mismatches = find_mismatches_by_allowed_values(df, allowed_values_dict)
        st.write("Mismatches based on allowed values:")
        for col, indices in mismatches.items():
            if indices:
                st.write(f"Column: {col}")
                st.dataframe(df.loc[indices])
    #button to perform automatic cleaning
    if st.button("Perform automatic cleaning"):
        df = automatic_cleaning(df,missing)
        st.session_state.df = df
        st.success("Automatic cleaning performed successfully!")
        st.dataframe(df.head(5))
