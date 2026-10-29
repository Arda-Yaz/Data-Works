"""One monthly fulfillment workflow backed by versioned, owned execution state."""

from collections import Counter
from itertools import islice
import hashlib
import json

import pandas as pd
import streamlit as st

from backend.diagnostics import log_event, log_failure
from backend.exports import export_csv
from backend.fulfillment import MAX_PLAN_BYTES, dumps_plan, ingest_csv, loads_plan, plan_digest, revise_plan
from backend.fulfillment_state import MAX_HISTORY
from backend import resources
from backend.state import StateManager
from frontend.project_status import show_project_download_status


_COLUMNS = ["order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status"]
_PREVIEW_ROWS = 100
_NO_STATUS = "Choose a status…"
_REASON_LABELS = {
    "missing_order_id": "Missing order ID", "missing_line_id": "Missing line ID",
    "identifier_whitespace": "Whitespace around an identifier",
    "missing_order_date": "Missing date", "invalid_order_date": "Invalid date",
    "missing_sku": "Missing SKU", "missing_quantity": "Missing quantity",
    "invalid_quantity": "Quantity is not a whole number",
    "quantity_out_of_range": "Quantity outside the supported range",
    "nonpositive_quantity": "Quantity is zero or negative",
    "missing_fulfillment_status": "Missing fulfillment status",
    "exact_duplicate": "Exact duplicate line",
}


def _next_action(workspace):
    if workspace["source"] is None:
        return "Next: load this month's order-line CSV, then review the reusable plan."
    attempt = workspace["last_attempt"]
    if attempt is None:
        return "Next: review the source and plan, then run the fulfillment plan for these inputs."
    if attempt["status"] == "blocked":
        return "Next: correct the file's columns using the issues below, load the corrected CSV, and run again."
    if attempt["status"] == "decision_required":
        return "Next: review the decisions below. Explicitly map an unseen status or correct conflicting records in the source, then run again."
    if attempt["status"] == "failed":
        return "Next: review the failed stage below and retry. This attempt committed no result."
    return "Next: review quarantine and report totals, download the outputs, and retain a project archive."


def _show_workspace_status(workspace):
    st.subheader("Workspace status")
    source = workspace["source"]
    plan = workspace["draft_plan"]
    latest = workspace["last_attempt"]
    committed = workspace["committed"]
    columns = st.columns(4)
    columns[0].caption("Current source")
    columns[0].text(f"{len(source['rows']):,} lines · {source['source_id'][:12]}" if source else "No source loaded")
    columns[1].caption("Current plan")
    columns[1].text(f"Revision {plan['revision']} · {plan['plan_id'][:12]}")
    columns[2].caption("Latest attempt for these inputs")
    columns[2].text(latest["status"].replace("_", " ") if latest else "Not run")
    columns[3].caption("Successful result")
    if committed:
        label = "Matches current inputs" if StateManager.fulfillment_result_is_current() else "Earlier source or plan"
        columns[3].text(f"{label} · {committed['attempt']['run_id'][:12]}")
    else:
        columns[3].text("No successful result")
    st.info(_next_action(workspace))
    show_project_download_status()
    st.caption(
        f"Undo retains {len(workspace['history'])} of at most {MAX_HISTORY} checkpoints within "
        f"{resources.MAX_HISTORY_BYTES / 1024**2:g} MiB. Oldest checkpoints are released first; "
        "a large change may leave no undo. Retain a project download before replacing work you need."
    )
    notice = st.session_state.get("fulfillment_change_notice")
    if notice and notice["revision"] == st.session_state.project_revision:
        if notice["pruned"]:
            st.warning(notice["message"])
        else:
            st.caption(notice["message"])
    error = st.session_state.get("fulfillment_decision_error")
    if error and error["revision"] == st.session_state.project_revision:
        st.error(error["message"])


def _note_change(message, previous_history_count, *, checkpoint=True):
    """Explain actual retention after a transition, including a skipped checkpoint."""
    remaining = len(st.session_state.fulfillment["history"])
    pruned = max(0, previous_history_count + int(checkpoint) - remaining)
    if pruned:
        message += f" Undo budget released {pruned} checkpoint(s); {remaining} remain. Retain a downloaded project for recovery beyond undo."
    st.session_state.fulfillment_change_notice = {
        "revision": st.session_state.project_revision, "message": message, "pruned": pruned,
    }


def _display_token(value):
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "…"


def _show_issue_guidance(attempt):
    for issue in attempt["issues"][:10]:
        code = issue["code"]
        if code == "missing_required_column":
            st.text("Missing required columns: " + ", ".join(issue["columns"]))
            st.caption("Export the required fields from the order system, then load the corrected CSV.")
        elif code == "unexpected_column":
            st.text("Unexpected columns: " + ", ".join(issue["columns"]))
            st.caption("Review the changed export schema. Supply the agreed six-column file; extra fields are not silently ignored.")
        elif code == "unseen_fulfillment_status":
            st.text(f"Unmapped status {_display_token(issue['normalized_value'])}: {len(issue['source_records']):,} source line(s).")
        elif code == "conflicting_duplicate":
            st.text(f"Conflicting duplicate: order {_display_token(issue['key'][0])}, line {_display_token(issue['key'][1])}.")
            st.text(f"Source records: {issue['source_records'][:_PREVIEW_ROWS]}. Differing fields: {', '.join(issue['differing_columns'])}.")
            st.caption("Confirm the correct values in the order system, correct the source export, and load it again. No winner has been selected.")
        elif code == "execution_failed":
            st.text(f"Failed stage: {issue['stage']}. Error type: {issue['error_type']}.")
    if len(attempt["issues"]) > 10:
        st.caption("Guidance shows the first 10 issues. Expand the issue evidence below for more references; the project retains every issue.")


def _show_status_decision(workspace):
    attempt = workspace["last_attempt"]
    if attempt is None or attempt["status"] != "decision_required":
        return
    unknowns = [issue["normalized_value"] for issue in attempt["issues"]
                if issue["code"] == "unseen_fulfillment_status"]
    if not unknowns:
        return
    # Run identity binds the widgets to this source and plan. A target choice
    # for another run or another source status cannot become a new decision.
    context = attempt["run_id"]
    st.write("Resolve an unseen status")
    st.caption("Choose its business meaning explicitly. Applying one mapping creates a plan revision; it does not run the file or change the previous result.")
    if len(unknowns) > _PREVIEW_ROWS:
        st.caption(f"Choosing among the first {_PREVIEW_ROWS} of {len(unknowns):,} unmapped statuses. Run after a decision to review remaining values, or use the advanced mapping editor.")
    status = st.selectbox("Source status to review", unknowns[:_PREVIEW_ROWS],
                          format_func=_display_token, key=f"fulfillment_decision_status_{context}")
    status_key = hashlib.sha256(status.encode("utf-8")).hexdigest()
    target = st.selectbox("Reviewed fulfillment meaning", [_NO_STATUS, "fulfilled", "pending", "cancelled"],
                          key=f"fulfillment_decision_target_{context}_{status_key}")
    st.button("Apply reviewed status", disabled=target == _NO_STATUS,
              key=f"fulfillment_decision_apply_{context}_{status_key}",
              on_click=_apply_reviewed_status, args=(context, status, f"fulfillment_decision_target_{context}_{status_key}"))


def _apply_reviewed_status(run_id, status, target_key):
    """Commit a reviewed decision before rendering the next workspace view."""
    workspace = st.session_state.fulfillment
    attempt = workspace["last_attempt"]
    target = st.session_state.get(target_key)
    is_current = (attempt is not None and attempt["run_id"] == run_id
                  and attempt["status"] == "decision_required"
                  and any(issue["code"] == "unseen_fulfillment_status"
                          and issue["normalized_value"] == status for issue in attempt["issues"]))
    if not is_current or target not in ("fulfilled", "pending", "cancelled"):
        st.session_state.fulfillment_decision_error = {
            "revision": st.session_state.project_revision,
            "message": "This decision no longer matches the current attempt. Review the current inputs and run again.",
        }
        return
    try:
        mapping = dict(workspace["draft_plan"]["steps"][1]["status_mapping"])
        mapping[status] = target
        previous_history_count = len(workspace["history"])
        StateManager.set_fulfillment_plan(revise_plan(workspace["draft_plan"], status_mapping=mapping))
    except Exception as exc:
        log_failure("fulfillment_plan_edit_failed", exc)
        st.session_state.fulfillment_decision_error = {
            "revision": st.session_state.project_revision,
            "message": "Mapping was not applied; the previous plan is unchanged. Review the plan and try again.",
        }
    else:
        _note_change("Reviewed status mapping applied. Run the revised plan to produce a new result.", previous_history_count)


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
        st.warning("A decision is required. Review the issues, revise the source or saved status mapping explicitly, then run again. This attempt committed no result.")
    elif status == "blocked":
        st.error("The current file is incompatible with the plan. This attempt committed no result.")
    else:
        st.error("The attempt failed before commitment. Review the stage and error type below before retrying.")
    if attempt["issues"]:
        _show_issue_guidance(attempt)
        with st.expander("Latest attempt issues"):
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
    st.caption("Report totals and fulfillment download buttons belong to this successful run, including when the latest attempt did not complete.")
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
    st.caption(f"Showing at most {_PREVIEW_ROWS} of {len(attempt['accepted']):,} accepted lines. The table toolbar exports only this preview. Use Download accepted CSV below for every accepted line in this result.")
    st.dataframe(_accepted_frame(attempt["accepted"][:_PREVIEW_ROWS]), hide_index=True, width="stretch")
    st.write("Quarantined lines with original values")
    counts = Counter(reason for record in attempt["rejected"] for reason in record["reasons"])
    reason = st.selectbox(
        "Inspect quarantine reason", ["All reasons", *sorted(counts)],
        format_func=lambda value: value if value == "All reasons" else f"{_REASON_LABELS.get(value, value)} ({counts[value]:,} lines)",
        key=f"fulfillment_rejection_reason_{attempt['run_id']}",
    )
    matching = len(attempt["rejected"]) if reason == "All reasons" else counts[reason]
    preview = list(islice((record for record in attempt["rejected"]
                           if reason == "All reasons" or reason in record["reasons"]), _PREVIEW_ROWS))
    st.caption(f"Showing {len(preview):,} of {matching:,} matching quarantined lines ({len(attempt['rejected']):,} total). This filter changes only the preview. The table toolbar exports this filtered preview; use Download quarantine CSV below for the full successful run. Report totals, source-row references, and all reasons are retained.")
    st.dataframe(_rejected_frame(preview), hide_index=True, width="stretch")

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
    _show_workspace_status(st.session_state.fulfillment)
    st.subheader("Monthly source file")
    uploaded = st.file_uploader("Choose an order-line CSV", type=["csv"], key="fulfillment_source_file")
    options = st.columns(2)
    encoding = options[0].text_input("Source text encoding", value="utf-8-sig", key="fulfillment_encoding")
    delimiter = options[1].selectbox("Source delimiter", [",", ";", "\t", "|"],
                                     format_func=lambda item: "Tab" if item == "\t" else item, key="fulfillment_delimiter")
    if st.button("Load fulfillment source", disabled=uploaded is None, key="fulfillment_load"):
        try:
            with st.spinner("Reading and validating the source…"):
                source = ingest_csv(uploaded.getvalue(), encoding=encoding, delimiter=delimiter)
                previous_history_count = len(st.session_state.fulfillment["history"])
                changed = StateManager.set_fulfillment_source(source)
        except Exception as exc:
            log_failure("fulfillment_source_failed", exc)
            st.error(f"Source was not loaded; the previous source is unchanged. {exc}")
        else:
            log_event("fulfillment_source_loaded", rows=len(source["rows"]), columns=len(source["columns"]))
            _note_change("Source loaded. Review the plan and run it for this file." if changed else "This source is already loaded.",
                         previous_history_count, checkpoint=changed)
            st.rerun()

    workspace = st.session_state.fulfillment
    source = workspace["source"]
    if source is not None:
        st.caption(f"Current source: {len(source['rows']):,} lines · Identity {source['source_id'][:12]}. Preview shows at most {_PREVIEW_ROWS} source lines and does not change execution scope.")
        with st.expander("Preview original source lines"):
            st.caption("The table toolbar exports only this source preview. The project archive retains the full original source.")
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
            previous_history_count = len(workspace["history"])
            changed = StateManager.set_fulfillment_plan(loads_plan(plan_file.getvalue()))
        except Exception as exc:
            log_failure("fulfillment_plan_open_failed", exc)
            st.error(f"Plan was not opened; the previous plan is unchanged. {exc}")
        else:
            _note_change("Saved plan opened. Run it for the current source." if changed else "This saved plan is already open.",
                         previous_history_count, checkpoint=changed)
            st.rerun()
    with st.expander("Review status mapping"):
        mapping_rows = list(islice(plan["steps"][1]["status_mapping"].items(), _PREVIEW_ROWS))
        st.dataframe(pd.DataFrame(mapping_rows, columns=["Source status (trimmed and lowercase)", "Fulfillment meaning"]),
                     hide_index=True, width="stretch")
        st.caption(f"Showing at most {_PREVIEW_ROWS} mappings. The table toolbar exports this preview; the downloaded plan and advanced JSON editor contain the full mapping.")
        st.write("Advanced mapping editor")
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
                    previous_history_count = len(workspace["history"])
                    StateManager.set_fulfillment_plan(candidate)
                    _note_change("Status mapping applied. Run the revised plan to produce a new result.", previous_history_count)
                    st.rerun()
            except Exception as exc:
                log_failure("fulfillment_plan_edit_failed", exc)
                st.error(f"Mapping was not applied; the previous plan is unchanged. {exc}")
    st.caption("Dates must be YYYY-MM-DD or YYYY/MM/DD. Quantities must be positive whole numbers; quantities are never filled. Exact duplicate lines keep the first source record; cancelled lines remain accepted.")
    st.subheader("Run and review")
    buttons = st.columns(2)
    if buttons[0].button("Run fulfillment plan", disabled=source is None, type="primary", key="fulfillment_run"):
        try:
            previous_history_count = len(workspace["history"])
            with st.spinner("Running the plan across every source line…"):
                attempt = StateManager.run_fulfillment()
        except Exception as exc:
            log_failure("fulfillment_run_failed", exc)
            st.error("The attempt could not be recorded. The previous workspace is unchanged.")
        else:
            if attempt["status"] == "completed":
                log_event("fulfillment_completed", accepted=attempt["report"]["accepted_rows"], rejected=attempt["report"]["rejected_rows"])
                _note_change("Completed result committed for the current source and plan.", previous_history_count)
            st.rerun()
    if buttons[1].button("Undo fulfillment change", disabled=not workspace["history"], key="fulfillment_undo"):
        try:
            with st.spinner("Restoring the previous fulfillment checkpoint…"):
                changed = StateManager.undo_fulfillment()
        except Exception as exc:
            log_failure("fulfillment_undo_failed", exc)
            st.error("The checkpoint could not be restored. The current workspace is unchanged.")
        else:
            if changed:
                _note_change("Previous source, plan, attempt, and result restored together.", len(st.session_state.fulfillment["history"]), checkpoint=False)
                st.rerun()

    # Replace the conditional region as one block. Resolved decision widgets
    # must disappear even when no prior successful result follows them.
    with st.empty().container():
        st.subheader("Latest attempt")
        _show_attempt(workspace["last_attempt"])
        _show_status_decision(workspace)
        _show_result(workspace)
