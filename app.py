import streamlit as st

from backend.state import StateManager
from backend.access import require_access
from backend.diagnostics import configure_logging, log_failure

st.set_page_config(page_title="DataWorks", layout="wide")
configure_logging()
require_access()

# Initialise shared session state once
StateManager.init_session()

_PAGES_NEED_DATA = {"Data Cleaning", "EDA", "Visualization", "Machine Learning"}


def home_page():
    st.sidebar.title("DataWorks")

    all_pages = ["Upload", "Data Cleaning", "EDA", "Visualization", "Machine Learning", "Project & Exports"]
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

    if page in _PAGES_NEED_DATA and page != "Machine Learning" and st.session_state.df.shape[1] == 0:
        st.warning("No columns remain. Undo the last operation or clear the dataset on the Upload page.")
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
    elif page == "Project & Exports":
        from frontend.pages.data_project import data_project
        data_project()


if __name__ == "__main__":
    try:
        home_page()
    except Exception as exc:
        log_failure("page_failed", exc)
        st.error("This page could not finish. Try a smaller operation or return to Upload; the server diagnostic log records the error type.")
