import copy
import re

import pandas as pd
import streamlit as st

from backend.exports import export_csv
from backend.operations import export_recipe, replay_operations
from backend.project_io import dumps_project, encode_value, loads_project
from backend.state import StateManager
from backend.diagnostics import log_event, log_failure


def _filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-_")[:80] or "dataworks"


def _save_project_section(filename):
    st.subheader("Save this project")
    st.caption(
        "Includes source/working data, schema, transformation parameters, saved charts, completed experiment reports, "
        "and fulfillment source, plan, results, and quarantine evidence when present. "
        "Live fitted models are exported separately from Machine Learning."
    )
    with_history = st.checkbox("Include undo snapshots (larger file)", value=True, key="project_include_history")
    token = (st.session_state.project_revision, with_history)
    prepared = st.session_state.get("prepared_project_download")
    if prepared and prepared["token"] != token:
        st.session_state.pop("prepared_project_download", None)
        prepared = None
    if st.button("Prepare Project Download", key="project_prepare"):
        try:
            archive = dumps_project(StateManager.export_project(include_history=with_history))
            prepared = {"data": archive, "token": token}
            st.session_state.prepared_project_download = prepared
        except Exception as exc:
            log_failure("project_save_failed", exc)
            st.error(f"Could not prepare the project. {exc}")
    if prepared:
        st.download_button(
            "Download DataWorks Project", prepared["data"], file_name=f"{filename}.dataworks",
            mime="application/zip", key="project_download", on_click="ignore",
        )
        st.caption(f"Prepared {len(prepared['data']):,} bytes. Retain the download to reopen this work; new changes require a fresh download.")


def data_project():
    st.title("Project & Exports")
    st.subheader("Open a project")
    st.caption("Opening a project replaces this session's dataset, history, charts, experiment reports, and fulfillment workspace.")
    uploaded = st.file_uploader("Choose a DataWorks project", type=["dataworks"], key="project_file")
    if st.button("Open Project", key="project_open", disabled=uploaded is None):
        try:
            payload = loads_project(uploaded.getvalue())
            StateManager.restore_project(payload)
        except Exception as exc:
            log_failure("project_open_failed", exc)
            st.error(f"Project was not opened; the current session is unchanged. {exc}")
        else:
            opened_df = st.session_state.df
            fulfillment_source = st.session_state.fulfillment.get("source")
            rows = len(opened_df) if opened_df is not None else len(fulfillment_source["rows"]) if fulfillment_source else 0
            columns = opened_df.shape[1] if opened_df is not None else len(fulfillment_source["columns"]) if fulfillment_source else 0
            log_event("project_opened", rows=rows, columns=columns)
            st.toast("Project opened.")
            st.rerun()

    fulfillment = st.session_state.fulfillment
    if not StateManager.has_dataset() and fulfillment["source"] is None and fulfillment["committed"] is None:
        st.info("Upload a dataset, load an Order Fulfillment source, or open a project to prepare downloads.")
        return
    name = st.text_input("Project name", value=st.session_state.project_name, max_chars=200, key="project_name_input")
    try:
        StateManager.set_project_name(name)
    except ValueError as exc:
        st.error(str(exc))
        return
    filename = _filename(name)
    df = st.session_state.df
    if df is None:
        st.caption("Fulfillment project. Review the source, saved plan, accepted lines, and quarantined lines on Order Fulfillment.")
        _save_project_section(filename)
        return
    st.caption(f"Working dataset: {len(df)} rows, {len(df.columns)} columns · State {st.session_state.data_version[:8]}")
    with st.expander("Session resource usage"):
        st.json(StateManager.resource_summary())
        st.caption("Undo snapshots and experiments are bounded by count and estimated memory; oldest entries are released when their budget is reached.")

    st.subheader("Download cleaned data")
    csv_mode = st.radio(
        "CSV text handling", ["spreadsheet_safe", "raw"],
        format_func=lambda value: "Spreadsheet-safe text" if value == "spreadsheet_safe" else "Raw values",
        horizontal=True, key="export_csv_mode",
    )
    include_index = st.checkbox("Include row index in CSV", value=False, key="export_csv_index")
    st.caption(
        "Spreadsheet-safe CSV prefixes formula-like text with an apostrophe. Raw CSV preserves that text; "
        "spreadsheet programs may interpret it as formulas. Project files preserve data types and schema."
    )
    exported = df.copy(deep=True)
    st.download_button(
        "Download cleaned CSV",
        lambda: export_csv(exported, spreadsheet_safe=csv_mode == "spreadsheet_safe", include_index=include_index),
        file_name=f"{filename}-cleaned.csv", mime="text/csv", key="export_cleaned_csv", on_click="ignore",
    )

    _save_project_section(filename)

    st.subheader("Transformation history")
    operations = st.session_state.operation_log
    if operations:
        rows = []
        for operation in operations:
            effect = operation.get("effects", {})
            rows.append({
                "Operation": operation["name"], "Type": operation["kind"],
                "Recorded UTC": operation.get("timestamp", ""),
                "Replayable": operation.get("replayable", False),
                "Rows before": effect.get("rows_before"), "Rows after": effect.get("rows_after"),
                "Columns before": effect.get("columns_before"), "Columns after": effect.get("columns_after"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
        with st.expander("Exact operation parameters"):
            for operation in operations:
                st.write(operation["name"])
                st.json(encode_value({"kind": operation["kind"], "parameters": operation["parameters"]}))
    else:
        st.info("No transformations have been applied to the original upload.")
    replayable = all(operation.get("replayable", False) for operation in operations)
    if not replayable:
        st.warning("Some older or custom changes have no complete recipe. They remain in the saved project, but this history cannot be fully replayed.")
    original = st.session_state.original_df
    original_metadata = copy.deepcopy(st.session_state.original_column_meta)
    records = copy.deepcopy(operations)
    st.download_button(
        "Download transformation recipe", lambda: export_recipe(original, original_metadata, records),
        file_name=f"{filename}-recipe.json", mime="application/json", key="project_recipe", on_click="ignore",
    )
    st.caption("The recipe identifies its original source table; the DataWorks project also contains that source data.")
    if st.button("Verify Transformation History", key="project_verify", disabled=not replayable):
        try:
            replayed = replay_operations(original, original_metadata, records)
            matches = (
                replayed["df"].equals(st.session_state.df)
                and replayed["column_meta"] == st.session_state.column_meta
                and replayed["schema_confirmed"] == st.session_state.schema_confirmed
                and replayed["statistical_operations"] == st.session_state.statistical_operations
            )
        except Exception as exc:
            st.error(f"History could not be replayed. {exc}")
        else:
            if matches:
                st.success("The recorded parameters reproduce the working data and schema exactly.")
            else:
                st.error("The recorded parameters do not reproduce the current state. Save the project to preserve its data; this recipe is not a complete reconstruction.")
