"""Shared, honest status for downloadable project archives."""

import streamlit as st


def show_project_download_status():
    """Describe preparation only; browser download receipt is not observable."""
    token = (st.session_state.get("project_revision", 0),
             st.session_state.get("project_include_history",
                                  st.session_state.get("_project_history_preference", True)))
    prepared = st.session_state.get("prepared_project_download")
    is_current = bool(prepared and prepared.get("token") == token)
    if is_current:
        st.caption(
            "Current archive prepared. Download and retain the .dataworks file from Project & Exports "
            "to reopen this work. DataWorks cannot confirm that your browser retained the download."
        )
    else:
        st.caption(
            "Current session only: prepare and download a .dataworks archive from Project & Exports "
            "to retain these changes. This session is not autosaved."
        )
    return is_current
