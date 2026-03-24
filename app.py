import streamlit as st

from frontend.pages.data_preview import preview


def home_page():
    #Choose Page
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a Page", ["Preview","Data Cleaning","Data Visualization",
                                                  "Statistical Tests", "Machine Models", "Model Guide"])

    if page == "Preview":
        preview()
    if page == "Data Cleaning":
        from frontend.pages.data_cleaning import data_cleaning
        data_cleaning()

if __name__ == "__main__":
    home_page()