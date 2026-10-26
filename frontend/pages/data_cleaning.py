import copy

import pandas as pd
import streamlit as st

from backend.cleaning import (
    build_missing_candidate, build_outlier_candidate, find_mismatches_by_allowed_values,
    get_duplicate_summary, get_missing_strategies, get_missing_summary,
    get_outlier_columns, outlier_summary, remove_duplicates,
)
from backend.cleaning_preview import describe_cleaning_preview
from backend.operations import apply_recipe_operation
from backend.state import StateManager
from backend.text_transform import get_text_columns
from backend.diagnostics import log_event, log_failure
from frontend.components.widgets import before_after_viewer


_OUTLIER_METHODS = ["iqr", "zscore"]
_OUTLIER_ACTIONS = ["ignore", "cap", "replace_nan", "remove"]


def _field_key(prefix, column):
    suffix = column if isinstance(column, str) else f"{type(column).__name__}:{column!r}"
    return f"{prefix}_{suffix}"


def _preview_operation(
    kind, controls, build, *, preview_label, preview_key, apply_label, apply_key,
    operation_name, learns_from_data=False, column_meta=None, operation=None,
):
    """Preview once, and apply only that candidate while its source/controls match."""
    df = st.session_state.df
    revision = st.session_state.dataset_revision
    entry = st.session_state.cleaning_previews.get(kind)
    if entry and (entry["revision"] != revision or entry["controls"] != controls):
        st.session_state.cleaning_previews = {}
        entry = None
        st.caption("Choices changed. Preview the current operation before applying it.")

    if st.button(preview_label, key=preview_key):
        st.session_state.cleaning_previews = {}
        entry = None
        try:
            positional = df.reset_index(drop=True)
            if operation is None:
                candidate = build(positional)
                metadata = copy.deepcopy(column_meta)
            else:
                candidate, metadata, _, _ = apply_recipe_operation(positional, st.session_state.column_meta, operation)
            report = describe_cleaning_preview(df, candidate)
            entry = {
                "report": report, "controls": copy.deepcopy(controls), "revision": revision,
                "column_meta": metadata, "operation_name": operation_name,
                "learns_from_data": learns_from_data,
                "operation": copy.deepcopy(operation),
            }
            st.session_state.cleaning_previews = {kind: entry}
        except Exception as exc:
            log_failure("cleaning_preview_failed", exc)
            st.error(f"Could not preview changes. {exc}")

    fresh = entry is not None and entry["revision"] == st.session_state.dataset_revision and entry["controls"] == controls
    changed = False
    if fresh:
        report = entry["report"]
        summary = report["summary"]
        changed = not report["candidate"].equals(df) or (
            entry["column_meta"] is not None and entry["column_meta"] != st.session_state.column_meta
        )
        st.subheader("Operation Preview")
        percentage = 100 * summary["rows_removed"] / max(summary["rows_before"], 1)
        st.write(
            f"Rows removed: {summary['rows_removed']} ({percentage:.1f}%) | "
            f"Columns removed: {summary['columns_removed']} | "
            f"Retained values edited: {summary['changed_cells']}"
        )
        st.caption(
            f"Missing values filled: {summary['filled_missing']} | "
            f"New missing values: {summary['introduced_missing']}. "
            "Row counts show the combined effect across all selected columns."
        )
        if report["dropped_columns"]:
            st.warning("Columns to remove: " + ", ".join(map(str, report["dropped_columns"])))
        if not report["column_changes"].empty:
            st.dataframe(report["column_changes"], width="stretch")
        st.caption("Before/after tables show corresponding affected rows, up to 20. Removed rows have no after entry.")
        before_after_viewer(report["before_sample"], report["after_sample"], summary=summary)
        if kind == "mismatch" and summary["rows_removed"] == 0:
            st.success("No mismatches found.")
        if not changed:
            st.info("No changes would be made.")
        elif report["candidate"].equals(df):
            st.info("Only the allowed-value rules will be updated.")

    if st.button(apply_label, key=apply_key, disabled=not (fresh and changed)):
        # Recheck server-side as well as disabling the button in the interface.
        current = st.session_state.cleaning_previews.get(kind)
        if not fresh or current is not entry or not changed:
            st.error("Preview the current choices before applying changes.")
            return
        try:
            applied = StateManager.apply_operation(
                entry["operation_name"], entry["report"]["candidate"],
                column_meta=entry["column_meta"], learns_from_data=entry["learns_from_data"],
                operation=entry.get("operation") or {
                    "kind": {"missing": "missing", "dupes": "duplicates", "outliers": "outliers", "mismatch": "mismatches"}[kind],
                    "parameters": {
                        "missing": {"strategies": controls.get("strategies", {}), "fill_values": controls.get("fills", {})},
                        "dupes": {"subset": controls.get("subset") or None, "keep": controls.get("keep", "first")},
                        "outliers": {"actions": controls.get("actions", {}), "include_protected": controls.get("include_protected", False)},
                        "mismatch": {"allowed_values": {
                            column: item["allowed_values"]
                            for column, item in (entry["column_meta"] or {}).items()
                        }},
                    }[kind],
                },
            )
        except Exception as exc:
            log_failure("cleaning_apply_failed", exc)
            st.error(f"Changes were not applied. {exc}")
        else:
            st.session_state.cleaning_previews = {}
            st.toast("Previewed changes applied." if applied else "No changes were needed.")
            if applied:
                log_event("cleaning_applied", rows=len(st.session_state.df), columns=st.session_state.df.shape[1])
                st.rerun()


def data_cleaning():
    st.title("Data Cleaning")
    if not StateManager.has_dataset():
        st.warning("Please upload a dataset in the Upload page first.")
        return

    df = st.session_state.df
    metadata = st.session_state.column_meta
    if st.session_state.history and st.button("↩ Undo Last Operation"):
        StateManager.undo()
        st.rerun()
    st.dataframe(df.head(5))
    if df.shape[1] == 0:
        st.info("No columns remain. Undo the last operation or upload another dataset.")
        return
    st.caption(
        "Choose an operation, preview its combined effect, then apply the displayed result. "
        "A new preview replaces the previous preview. Whole-number columns become decimals only when needed."
    )
    tab_missing, tab_dupes, tab_outliers, tab_mismatch, tab_text = st.tabs(
        ["Missing Values", "Duplicates", "Outliers", "Mismatches", "Text & Categories"],
    )

    with tab_missing:
        summary = get_missing_summary(df, metadata)
        if summary.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(summary)
            st.caption(
                "Missing percentage alone cannot determine the right treatment. "
                "Consider why values are missing and the column's purpose. Keep is the default."
            )
            strategies, fills = {}, {}
            for column in summary.index:
                options = get_missing_strategies(df[column])
                key = _field_key("ms", column)
                if key in st.session_state and st.session_state[key] not in options:
                    del st.session_state[key]
                c1, c2 = st.columns([2, 1])
                with c1:
                    strategies[column] = st.selectbox(
                        f"Strategy for **{column}**", options,
                        index=options.index("keep"), key=key,
                    )
                with c2:
                    if strategies[column] == "constant":
                        fills[column] = st.text_input(
                            f"Fill value for {column}", key=_field_key("fv", column),
                            help="Numbers are parsed as numbers; booleans accept true/false. "
                                 "Dates use ISO format. Naive dates use the column's timezone.",
                        )
            _preview_operation(
                "missing", {"strategies": strategies, "fills": fills},
                lambda base: build_missing_candidate(base, strategies, fills),
                preview_label="Preview Missing Value Changes", preview_key="preview_missing",
                apply_label="Apply Missing Value Strategies", apply_key="apply_missing",
                operation_name="Handle missing values",
                learns_from_data=any(
                    strategy in ("mean", "median", "mode", "interpolate")
                    for strategy in strategies.values()
                ),
            )

    with tab_dupes:
        subset = st.multiselect(
            "Check duplicates across specific columns (leave empty for all)",
            df.columns.tolist(), key="dup_subset",
        )
        keep = st.radio("Keep", ["first", "last", False], horizontal=True, key="dup_keep")
        count, sample = get_duplicate_summary(df, subset=subset or None)
        st.write(f"**{count}** rows belong to duplicate groups.")
        if count:
            with st.expander("Show duplicate-group sample"):
                st.dataframe(sample)
        _preview_operation(
            "dupes", {"subset": subset, "keep": keep},
            lambda base: remove_duplicates(base, subset=subset or None, keep=keep)[0],
            preview_label="Preview Duplicate Removal", preview_key="preview_dupes",
            apply_label="Remove Duplicates", apply_key="apply_dupes",
            operation_name="Remove duplicates",
        )

    with tab_outliers:
        protected = st.checkbox(
            "Include IDs or numeric columns marked categorical",
            value=False, key="outlier_include_protected",
            help="These columns may contain identifiers or category codes rather than measured quantities. "
                 "Enable this only when the selected columns are suitable for statistical outlier handling.",
        )
        columns = get_outlier_columns(df, metadata, include_protected=protected)
        if "outlier_columns" in st.session_state:
            st.session_state.outlier_columns = [
                column for column in st.session_state.outlier_columns if column in columns
            ]
        selected = st.multiselect(
            "Columns to inspect for outliers", columns, default=columns, key="outlier_columns",
        )
        excluded = [column for column in get_outlier_columns(df, metadata, include_protected=True) if column not in columns]
        if excluded:
            st.caption("Excluded by schema: " + ", ".join(map(str, excluded)))
        st.caption(
            "Outliers can be valid observations. Actions default to ignore. "
            "IQR uses 1.5 × the interquartile range; z-score uses three sample standard deviations. "
            "Every bound is calculated from the same data before this operation."
        )
        if not selected:
            if "outliers" in st.session_state.cleaning_previews:
                st.session_state.cleaning_previews = {}
            st.info("Select numeric measurements to inspect. Protected numeric columns require explicit inclusion.")
        else:
            actions = {}
            for column in selected:
                c1, c2 = st.columns(2)
                with c1:
                    method = st.selectbox(
                        f"Method for **{column}**", _OUTLIER_METHODS, key=_field_key("om", column),
                    )
                with c2:
                    action = st.selectbox(
                        f"Action for **{column}**", _OUTLIER_ACTIONS, key=_field_key("oa", column),
                    )
                actions[column] = (method, action)
            try:
                st.dataframe(outlier_summary(df, {column: method for column, (method, _) in actions.items()}))
            except ValueError as exc:
                st.error(f"Could not inspect outliers. {exc}")
            _preview_operation(
                "outliers", {"actions": actions, "include_protected": protected},
                lambda base: build_outlier_candidate(base, actions),
                preview_label="Preview Outlier Changes", preview_key="preview_outliers",
                apply_label="Apply Outlier Handling", apply_key="apply_outliers",
                operation_name="Handle outliers",
                learns_from_data=any(action != "ignore" for _, action in actions.values()),
            )

    with tab_mismatch:
        candidate_meta = copy.deepcopy(metadata)
        allowed = {}
        st.caption(
            "Apply Allowed Values saves rules without changing rows. "
            "Preview Mismatches shows row removals using your current entries. "
            "Rules follow the column's type; dates use ISO format."
        )
        for column in df.columns:
            current = metadata[column].get("allowed_values")
            text = st.text_input(
                f"Allowed values for **{column}** (comma-separated)",
                value=", ".join(current) if current else "", key=_field_key("av", column),
            )
            values = [value.strip() for value in text.split(",") if value.strip()]
            candidate_meta[column]["allowed_values"] = values or None
            if values:
                allowed[column] = values
        if st.button("Apply Allowed Values", key="apply_allowed_values"):
            try:
                changed = StateManager.apply_operation(
                    "Update allowed values", df, column_meta=candidate_meta,
                    operation={"kind": "rules", "parameters": {
                        "allowed_values": {column: item["allowed_values"] for column, item in candidate_meta.items()},
                    }},
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.toast("Allowed values saved.")
                if changed:
                    st.rerun()

        def mismatch_candidate(base):
            mismatches = find_mismatches_by_allowed_values(base, allowed)
            removed = {position for positions in mismatches.values() for position in positions}
            return base.drop(index=list(removed))

        _preview_operation(
            "mismatch", {"allowed_values": allowed}, mismatch_candidate,
            preview_label="Preview Mismatches", preview_key="find_mismatch",
            apply_label="Remove Mismatch Rows", apply_key="apply_mismatch",
            operation_name="Remove mismatch rows", column_meta=candidate_meta,
        )

    with tab_text:
        _text_controls(df)


def _text_controls(df):
    columns = get_text_columns(df)
    if not columns:
        st.info("No text or category columns are available.")
        return
    mode = st.radio("Text operation", ["Normalize text", "Map categories"], horizontal=True, key="text_mode")
    st.caption("Changes affect only text values in the selected columns. Missing values stay missing. Matching allowed-value rules are updated too.")
    if mode == "Normalize text":
        selected = st.multiselect("Text columns", columns, key="text_columns")
        strip = st.checkbox("Trim leading and trailing whitespace", key="text_strip")
        collapse = st.checkbox("Collapse repeated whitespace", key="text_collapse")
        case = st.selectbox("Letter case", ["keep", "lower", "upper", "casefold"], key="text_case")
        form = st.selectbox("Unicode normalization", [None, "NFC", "NFKC"], format_func=lambda value: value or "Keep unchanged", help="NFC combines equivalent characters. NFKC also folds compatibility forms such as full-width letters.", key="text_unicode")
        if not selected:
            st.info("Select the columns to normalize.")
            return
        operation = {"kind": "text_normalize", "parameters": {
            "columns": selected, "strip": strip, "collapse_whitespace": collapse,
            "case": case, "unicode_form": form,
        }}
        name = "Normalize text"
    else:
        column = st.selectbox("Category column", columns, key="text_category_column")
        st.caption("Exact replacements run simultaneously; unmatched values stay unchanged. A blank replacement produces empty text.")
        pairs = st.data_editor(
            pd.DataFrame({"Original": pd.Series(dtype=str), "Replacement": pd.Series(dtype=str)}),
            num_rows="dynamic", hide_index=True, key="text_mapping",
            column_config={"Original": st.column_config.TextColumn(required=True), "Replacement": st.column_config.TextColumn()},
        )
        mapping = {}
        for source, replacement in pairs.itertuples(index=False, name=None):
            if pd.isna(source) and pd.isna(replacement):
                continue
            if not isinstance(source, str) or source in mapping:
                st.error("Each original category must be text and appear only once.")
                return
            replacement = "" if pd.isna(replacement) else replacement
            if not isinstance(replacement, str):
                st.error("Replacement categories must be text.")
                return
            mapping[source] = replacement
        if not mapping:
            st.info("Add the categories and their replacements.")
            return
        operation = {"kind": "category_map", "parameters": {"column": column, "mapping": mapping}}
        name = "Map categories"
    _preview_operation(
        "text", operation, None, operation=operation,
        preview_label="Preview Text Changes", preview_key="text_preview",
        apply_label="Apply Text Changes", apply_key="text_apply", operation_name=name,
    )
