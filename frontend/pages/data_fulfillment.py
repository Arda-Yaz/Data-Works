"""One monthly fulfillment workflow backed by versioned, owned execution state."""

import json

import pandas as pd
import streamlit as st

from backend.diagnostics import log_event, log_failure
from backend.exports import export_csv
from backend.fulfillment import MAX_PLAN_BYTES, dumps_plan, ingest_csv, loads_plan, plan_digest, revise_plan
from backend.state import StateManager


_COLUMNS = ["order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status"]
_PREVIEW_ROWS = 100


def _mapping_from_json(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_PLAN_BYTES:
        raise ValueError("The status mapping must be JSON within the 1 MiB limit.")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate status keys are not allowed.")
            result[key] = value
        return result

    def finite_only(value):
        raise ValueError("Nonfinite JSON values are not allowed.")

    try:
        mapping = json.loads(text, object_pairs_hook=unique_keys, parse_constant=finite_only)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("Use a JSON object with unique status keys and finite values.") from exc
    if not isinstance(mapping, dict):
        raise ValueError("The status mapping must be a JSON object.")
    return mapping


def _accepted_frame(rows):
    # Object storage avoids converting large integral quantities through floats.
    return pd.DataFrame([item["values"] for item in rows], columns=_COLUMNS, dtype=object)


def _rejected_frame(rows):
    columns = ["source_record", "row_id", "reason_codes", "validation_details",
               "duplicate_of_source_record", "duplicate_of_row_id", *_COLUMNS]
    records = []
    for item in rows:
        records.append({
            "source_record": item["source_record"], "row_id": item["row_id"],
            "reason_codes": json.dumps(item["reasons"], ensure_ascii=False),
            "validation_details": json.dumps(item["issues"], ensure_ascii=False),
            "duplicate_of_source_record": str(item["duplicate_of_source_record"]) if "duplicate_of_source_record" in item else "",
            "duplicate_of_row_id": item.get("duplicate_of_row_id", ""),
            **item["raw"],
        })
    return pd.DataFrame(records, columns=columns, dtype=object)


def _show_attempt(attempt):
    if attempt is None:
        st.info("The current source has not been run yet.")
        return
    status = attempt["status"]
    if status == "completed":
        report = attempt["report"]
        st.success(f"Latest attempt completed: {report['accepted_rows']:,} accepted lines and {report['rejected_rows']:,} quarantined lines.")
    elif status == "decision_required":
        st.warning("A decision is required. Review the issues, revise the source or saved status mapping explicitly, then run again. This attempt did not replace the last successful result.")
    elif status == "blocked":
        st.error("The current file is incompatible with the plan. This attempt did not replace the last successful result.")
    else:
        st.error("The attempt failed before commitment. The last successful result is retained. Review the stage and error type below before retrying.")
    if attempt["issues"]:
        with st.expander("Latest attempt issues", expanded=status != "completed"):
            preview = []
            for issue in attempt["issues"][:_PREVIEW_ROWS]:
                shown = dict(issue)
                if isinstance(shown.get("source_records"), list) and len(shown["source_records"]) > _PREVIEW_ROWS:
                    shown["source_record_count"] = len(shown["source_records"])
                    shown["source_records"] = shown["source_records"][:_PREVIEW_ROWS]
                preview.append(shown)
            st.caption(f"Showing at most {_PREVIEW_ROWS} of {len(attempt['issues']):,} issues and {_PREVIEW_ROWS} source-record references per issue. The project retains every issue.")
            st.json(preview)
    with st.expander("Latest attempt details"):
        st.json({key: attempt[key] for key in (
            "run_id", "source_id", "plan_id", "plan_revision", "plan_digest", "engine_version", "status",
        )})


def _show_result(workspace):
    committed = workspace["committed"]
    if committed is None:
        st.info("No successful fulfillment result has been committed yet.")
        return
    attempt = committed["attempt"]
    report = attempt["report"]
    st.subheader("Last successful result")
    if not StateManager.fulfillment_result_is_current():
        st.warning("This result belongs to an earlier source or plan. Run the current plan to update it.")
    if workspace["last_attempt"] and workspace["last_attempt"]["run_id"] != attempt["run_id"]:
        st.caption("The latest attempt and this successful result are different runs.")
    st.caption(
        f"Source {attempt['source_id'][:12]} · Plan revision {attempt['plan_revision']} · Run {attempt['run_id'][:12]}"
    )
    totals = st.columns(4)
    for column, label, value in zip(totals, ("Source lines", "Accepted lines", "Quarantined lines", "Accepted orders"),
                                    (report["source_rows"], report["accepted_rows"], report["rejected_rows"], report["accepted_distinct_orders"])):
        column.metric(label, str(value))
    st.caption(f"Accepted quantity: {report['accepted_quantity']:,}. Reason assignments: {report['rejection_reason_assignments']:,}; a line may have more than one reason.")
    statuses = []
    for status, count in report["accepted_line_counts_by_status"].items():
        statuses.append({"Status": status, "Accepted lines": str(count),
                         "Accepted quantity": str(report["accepted_quantity_by_status"][status])})
    st.dataframe(pd.DataFrame(statuses), hide_index=True, width="stretch")

    st.write("Accepted lines")
    st.caption(f"Showing at most {_PREVIEW_ROWS} of {len(attempt['accepted']):,} accepted lines. Downloads contain every line in this result.")
    st.dataframe(_accepted_frame(attempt["accepted"][:_PREVIEW_ROWS]), hide_index=True, width="stretch")
    st.write("Quarantined lines with original values")
    st.caption(f"Showing at most {_PREVIEW_ROWS} of {len(attempt['rejected']):,} quarantined lines. Source-row references and all reasons are retained.")
    st.dataframe(_rejected_frame(attempt["rejected"][:_PREVIEW_ROWS]), hide_index=True, width="stretch")

    mode = st.radio(
        "Fulfillment CSV text handling", ["spreadsheet_safe", "raw"],
        format_func=lambda value: "Spreadsheet-safe text" if value == "spreadsheet_safe" else "Raw values",
        horizontal=True, key="fulfillment_export_mode",
    )
    st.caption("Spreadsheet-safe CSV prefixes formula-like text with an apostrophe. Raw CSV preserves the source text; the project always retains original values and types.")
    # State transitions replace owned results. The deferred exporters read this
    # committed run and build new frames, so another full copy per rerun is wasteful.
    accepted = attempt["accepted"]
    rejected = attempt["rejected"]
    suffix = attempt["run_id"][:12]
    st.download_button(
        "Download accepted CSV", lambda: export_csv(_accepted_frame(accepted), spreadsheet_safe=mode == "spreadsheet_safe"),
        file_name=f"fulfillment-{suffix}-accepted.csv", mime="text/csv", key="fulfillment_accepted_download", on_click="ignore",
    )
    st.download_button(
        "Download quarantine CSV", lambda: export_csv(_rejected_frame(rejected), spreadsheet_safe=mode == "spreadsheet_safe"),
        file_name=f"fulfillment-{suffix}-quarantine.csv", mime="text/csv", key="fulfillment_rejected_download", on_click="ignore",
    )
    report_document = {key: attempt[key] for key in (
        "run_id", "source_id", "plan_id", "plan_revision", "plan_digest", "engine_version", "report",
    )}
    st.download_button(
        "Download fulfillment report", json.dumps(report_document, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"fulfillment-{suffix}-report.json", mime="application/json", key="fulfillment_report_download", on_click="ignore",
    )
    st.info("Use Project & Exports to download a .dataworks archive containing source evidence, the saved plan, run results, and optional undo history. Retain that file to reopen this work in a new session.")


def data_fulfillment():
    st.title("Order Fulfillment")
    st.caption("Prepare a monthly order-line file for fulfillment reporting. Identifiers remain text, invalid records are quarantined, and conflicting duplicates require a decision.")
    st.subheader("Monthly source file")
    uploaded = st.file_uploader("Choose an order-line CSV", type=["csv"], key="fulfillment_source_file")
    options = st.columns(2)
    encoding = options[0].text_input("Source text encoding", value="utf-8-sig", key="fulfillment_encoding")
    delimiter = options[1].selectbox("Source delimiter", [",", ";", "\t", "|"],
                                     format_func=lambda item: "Tab" if item == "\t" else item, key="fulfillment_delimiter")
    if st.button("Load fulfillment source", disabled=uploaded is None, key="fulfillment_load"):
        try:
            source = ingest_csv(uploaded.getvalue(), encoding=encoding, delimiter=delimiter)
            StateManager.set_fulfillment_source(source)
        except Exception as exc:
            log_failure("fulfillment_source_failed", exc)
            st.error(f"Source was not loaded; the previous source is unchanged. {exc}")
        else:
            log_event("fulfillment_source_loaded", rows=len(source["rows"]), columns=len(source["columns"]))
            st.rerun()

    workspace = st.session_state.fulfillment
    source = workspace["source"]
    if source is not None:
        st.caption(f"Current source: {len(source['rows']):,} lines · Identity {source['source_id'][:12]}. Preview shows at most {_PREVIEW_ROWS} source lines and does not change execution scope.")
        st.dataframe(pd.DataFrame(source["rows"][:_PREVIEW_ROWS], columns=source["columns"], dtype=object), hide_index=True, width="stretch")
    else:
        st.info("Load a CSV with order_id, line_id, order_date, sku, quantity, and fulfillment_status.")

    st.subheader("Reusable plan")
    plan = workspace["draft_plan"]
    digest = plan_digest(plan)
    st.caption(f"Plan revision {plan['revision']} · Identity {plan['plan_id'][:12]} · Digest {digest[:12]}")
    st.download_button("Download fulfillment plan", dumps_plan(plan), file_name="fulfillment-plan.json",
                       mime="application/json", key="fulfillment_plan_download", on_click="ignore")
    plan_file = st.file_uploader("Choose a saved fulfillment plan", type=["json"], key="fulfillment_plan_file")
    if st.button("Open fulfillment plan", disabled=plan_file is None, key="fulfillment_plan_open"):
        try:
            StateManager.set_fulfillment_plan(loads_plan(plan_file.getvalue()))
        except Exception as exc:
            log_failure("fulfillment_plan_open_failed", exc)
            st.error(f"Plan was not opened; the previous plan is unchanged. {exc}")
        else:
            st.rerun()
    with st.expander("Review status mapping"):
        st.caption("Keys are normalized source statuses; values must be fulfilled, pending, or cancelled. Applying a change creates a new plan revision. Unseen statuses prevent commitment until reviewed.")
        text = st.text_area("Status mapping as JSON", value=json.dumps(plan["steps"][1]["status_mapping"], ensure_ascii=False, indent=2),
                            key=f"fulfillment_mapping_{digest}", height=230)
        if st.button("Apply status mapping", key="fulfillment_mapping_apply"):
            try:
                mapping = _mapping_from_json(text)
                if mapping == plan["steps"][1]["status_mapping"]:
                    st.info("The status mapping is unchanged.")
                else:
                    candidate = revise_plan(plan, status_mapping=mapping)
                    StateManager.set_fulfillment_plan(candidate)
                    st.rerun()
            except Exception as exc:
                log_failure("fulfillment_plan_edit_failed", exc)
                st.error(f"Mapping was not applied; the previous plan is unchanged. {exc}")
    st.caption("Dates must be YYYY-MM-DD or YYYY/MM/DD. Quantities must be positive whole numbers; quantities are never filled. Exact duplicate lines keep the first source record; cancelled lines remain accepted.")
    buttons = st.columns(2)
    if buttons[0].button("Run fulfillment plan", disabled=source is None, type="primary", key="fulfillment_run"):
        try:
            attempt = StateManager.run_fulfillment()
        except Exception as exc:
            log_failure("fulfillment_run_failed", exc)
            st.error("The attempt could not be recorded. The previous workspace is unchanged.")
        else:
            if attempt["status"] == "completed":
                log_event("fulfillment_completed", accepted=attempt["report"]["accepted_rows"], rejected=attempt["report"]["rejected_rows"])
            st.rerun()
    if buttons[1].button("Undo fulfillment change", disabled=not workspace["history"], key="fulfillment_undo"):
        if StateManager.undo_fulfillment():
            st.rerun()

    st.subheader("Latest attempt")
    _show_attempt(workspace["last_attempt"])
    _show_result(workspace)
