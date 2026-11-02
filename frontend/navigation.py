"""Small, persistent navigation for the two independent data workspaces."""

import streamlit as st

from backend.state import StateManager


DATA_PAGES = ("Upload", "Data Cleaning", "EDA", "Visualization", "Machine Learning")
WORKSPACES = ("Data preparation", "Order Fulfillment")


def open_page(page):
    if page in DATA_PAGES and page != "Upload" and not StateManager.has_dataset():
        page = "Upload"
    st.session_state["_navigation_page"] = page
    if page in DATA_PAGES:
        st.session_state["_last_data_page"] = page
    if page != "Project & Exports":
        st.session_state["_last_workspace_page"] = page


def _change_workspace():
    if st.session_state["workspace"] == "Order Fulfillment":
        open_page("Order Fulfillment")
    else:
        open_page(st.session_state.get("_last_data_page", "Upload"))


def _page_button(label, page, key, *, disabled=False):
    st.sidebar.button(
        label, key=key, width="stretch", disabled=disabled,
        type="primary" if st.session_state["_navigation_page"] == page else "secondary",
        on_click=open_page, args=(page,),
    )


def show_navigation():
    st.session_state.setdefault("_navigation_page", "Upload")
    # A reopened fulfillment-only project may have replaced the generic dataset.
    # Normalize the route itself so loading another file stays on schema review.
    if st.session_state["_navigation_page"] in DATA_PAGES and not StateManager.has_dataset():
        open_page("Upload")
    st.sidebar.title("DataWorks")
    st.sidebar.caption("Prepare data. Understand the result.")
    st.sidebar.selectbox("Workspace", WORKSPACES, key="workspace", on_change=_change_workspace)

    page = st.session_state["_navigation_page"]
    has_data = StateManager.has_dataset()
    if st.session_state["workspace"] == "Data preparation":
        st.sidebar.caption("YOUR WORKFLOW")
        _page_button("1 · Load & review", "Upload", "nav_upload")
        _page_button("2 · Clean data", "Data Cleaning", "nav_clean", disabled=not has_data)
        _page_button("3 · Explore data", "EDA", "nav_explore", disabled=not has_data)
        with st.sidebar.expander("Charts & models", expanded=page in ("Visualization", "Machine Learning")):
            st.button("Build charts", key="nav_charts", width="stretch", disabled=not has_data,
                      type="primary" if page == "Visualization" else "secondary",
                      on_click=open_page, args=("Visualization",))
            st.button("Train models", key="nav_models", width="stretch", disabled=not has_data,
                      type="primary" if page == "Machine Learning" else "secondary",
                      on_click=open_page, args=("Machine Learning",))
        if has_data:
            frame = st.session_state.df
            st.sidebar.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns")
            st.sidebar.caption("Column types confirmed" if st.session_state.schema_confirmed else "Review column types before analysis")
        else:
            st.sidebar.caption("Load a file to unlock cleaning and analysis.")
    else:
        _page_button("Process monthly orders", "Order Fulfillment", "nav_fulfillment")
        st.sidebar.caption("A separate workspace for six-column order-line CSVs.")

    st.sidebar.divider()
    _page_button("Save, export & reopen", "Project & Exports", "nav_project")
    if page == "Project & Exports":
        _page_button("Back to workspace", st.session_state.get("_last_workspace_page", "Upload"), "nav_back")
    st.sidebar.caption("Work stays in this session until you download a project.")

    history = st.session_state.get("history", [])
    if history:
        with st.sidebar.expander("Dataset history"):
            st.subheader("Dataset history")
            for entry in reversed(history[-5:]):
                st.text(entry["name"])
            if st.button("↩ Undo dataset change", key="nav_dataset_undo", width="stretch"):
                if StateManager.undo():
                    st.rerun()
    return st.session_state["_navigation_page"]
