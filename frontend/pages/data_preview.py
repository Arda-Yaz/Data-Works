import pandas as pd 
import streamlit as st
import backend.preview as preview_mod        # avoid name clash
# alternatively:
# from backend.preview import dataset_overview, sanitize

def preview():
    st.title("Data Preview")
    st.write("Upload your dataset to preview its contents.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

    if "df" in st.session_state:
        df = st.session_state.df
        clean = preview_mod.sanitize(df)

        # basic info
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write("**Columns and dtypes:**")
        st.table(pd.DataFrame(df.dtypes, columns=["dtype"]))

        # let user pick number of rows to display; clamp to actual size
        max_rows = max(1, df.shape[0])
        n = st.sidebar.slider(
            "Number of rows to show",
            1,
            min(max_rows, 100),
            5 if max_rows >= 5 else 1,
            step=1,
        )
        st.dataframe(df.head(n))

        # statistics and missing values in expanders
        with st.expander("Summary statistics"):
            st.dataframe(df.describe(include="all").transpose())

        with st.expander("Missing values"):
            missing = df.isna().sum().to_frame("missing")
            missing["%"] = 100 * missing["missing"] / len(df)
            st.dataframe(missing)

        with st.expander("Guessed schema"):
            st.table(pd.DataFrame({
                'original dtype': df.dtypes,
                'guessed dtype': clean.dtypes
            }))

        # show overview from backend
        preview_mod.dataset_overview(df)

        # let user choose to accept the cleaned version, or keep the raw one
        if st.button("Use cleaned types"):
            st.session_state.df = clean
            st.rerun()

        if st.button("Clear dataset"):
            del st.session_state["df"]
            st.rerun()

        # let user edit the schema
        schema = (
            pd.DataFrame({
                "column": df.columns,
                "original": df.dtypes.astype(str),
                "guessed": clean.dtypes.astype(str),
            })
            .set_index("column")
        )
        # add an editable column initialised to the guess
        schema["type"] = schema["guessed"]

        edited = st.data_editor(
            schema,
            num_rows="dynamic",            # user can add/remove if you want
            width='stretch',
        )
        # `edited` is a DataFrame with whatever the user picked

        if st.button("Apply types"):
            # walk through the edited choices and cast accordingly
            for col, dtype in edited["type"].items():
                if dtype != str(df[col].dtype):
                    df[col] = df[col].astype(dtype, errors="ignore")
            st.session_state.df = df
            st.rerun()





