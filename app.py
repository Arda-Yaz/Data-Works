import streamlit as st

from backend.state import StateManager

st.set_page_config(page_title="DataWorks", layout="wide")

# Initialise shared session state once
StateManager.init_session()

_PAGES_NEED_DATA = {"Data Cleaning", "EDA", "Visualization", "Machine Learning"}


def home_page():
    st.sidebar.title("DataWorks")

    all_pages = ["Upload", "Data Cleaning", "EDA", "Visualization", "Machine Learning"]
    page = st.sidebar.selectbox("Navigate", all_pages)

    # Undo button + operation history in sidebar
    history = st.session_state.get("history", [])
    if history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("History")
        for entry in reversed(history[-5:]):
            st.sidebar.text(f"• {entry['name']}")
        if st.sidebar.button("↩ Undo Last"):
            if StateManager.undo():
                st.rerun()

    # Guard: pages that need a loaded dataset
    if page in _PAGES_NEED_DATA and not StateManager.has_dataset():
        _show_page("Upload")
        st.warning(f"⬆ Upload a dataset first to access **{page}**.")
        return

    _show_page(page)


def _show_page(page: str):
    if page == "Upload":
        from frontend.pages.data_upload import data_upload
        data_upload()
    elif page == "Data Cleaning":
        from frontend.pages.data_cleaning import data_cleaning
        data_cleaning()
    elif page == "EDA":
        from frontend.pages.data_eda import data_eda
        data_eda()
    elif page == "Visualization":
        from frontend.pages.data_viz import data_viz
        data_viz()
    elif page == "Machine Learning":
        from frontend.pages.data_ml import data_ml
        data_ml()


if __name__ == "__main__":
    home_page()