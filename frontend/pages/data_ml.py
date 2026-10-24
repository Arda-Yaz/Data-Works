import pandas as pd
import streamlit as st

from backend.ingestion import apply_inferred_dtypes
from backend.ml import (
    detect_problem_type, evaluate_model, get_available_models,
    plot_confusion_matrix, plot_feature_importance, plot_residuals, train_model,
)
from backend.preprocessing import auto_preprocess
from backend.state import StateManager


_SPLIT_LABELS = {
    "stratified": "Stratified random", "random": "Random",
    "group": "Grouped", "chronological": "Chronological",
}
_SOURCE_LABELS = {
    "working": "Working dataset", "original": "Original upload with current schema",
}


def _reset_invalid_selection(key, options):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def data_ml():
    st.title("Machine Learning")
    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return

    working = st.session_state.df
    metadata = st.session_state.column_meta
    sources = ["working", "original"] if st.session_state.original_df is not None else ["working"]
    statistical = st.session_state.statistical_operations
    _reset_invalid_selection("ml_data_source", sources)
    source = st.selectbox(
        "Modeling data", sources, index=1 if statistical and "original" in sources else 0,
        format_func=_SOURCE_LABELS.get, key="ml_data_source",
    )
    st.caption(
        "The original upload uses your current columns and types, before row cleaning or filling. "
        "ML feature fills, encodings, and scaling are fitted on training rows."
    )
    if statistical and source == "working":
        st.warning(
            "Earlier cleaning may have used statistics from the full dataset. Scores on this working dataset "
            "may be optimistic because test rows influenced cleaning. Use the original upload "
            "to skip those fitted cleaning values; working-data scores are exploratory."
        )
    try:
        if source == "original":
            original = st.session_state.original_df
            if any(column not in original.columns for column in working.columns):
                raise ValueError("The original upload does not contain every current column.")
            df = apply_inferred_dtypes(original.loc[:, working.columns], metadata)
        else:
            df = working
    except ValueError as exc:
        st.error(f"Could not prepare the original data with the current schema. {exc}")
        return

    eligible = [
        column for column in df.columns
        if metadata.get(column, {}).get("role") not in ("id", "datetime", "ignore")
    ]
    if not eligible:
        st.warning("Mark at least one column as a feature in Schema Review to choose a target.")
        return
    _reset_invalid_selection("ml_target", eligible)
    saved_target = st.session_state.target_column
    target = st.selectbox(
        "Select target column", eligible,
        index=eligible.index(saved_target) if saved_target in eligible else 0, key="ml_target",
    )
    if target is None:
        return
    StateManager.set_target(target)
    context = (source, target)
    if st.session_state.get("ml_target_context") != context:
        for key in (
            "ml_ptype", "ml_features", "ml_feature_intent", "ml_feature_scope", "ml_models",
            "ml_split_strategy", "ml_group_col", "ml_time_col",
        ):
            st.session_state.pop(key, None)
        st.session_state["ml_target_context"] = context

    auto_type = detect_problem_type(df[target])
    problem_type = st.radio(
        "Problem type", ["classification", "regression"],
        index=0 if auto_type == "classification" else 1,
        horizontal=True, key="ml_ptype",
    )
    strategies = ["stratified", "random", "group", "chronological"] if problem_type == "classification" else ["random", "group", "chronological"]
    _reset_invalid_selection("ml_split_strategy", strategies)
    strategy = st.selectbox(
        "Split method", strategies, format_func=_SPLIT_LABELS.get, key="ml_split_strategy",
    )
    group_col = time_col = None
    split_options = [column for column in df.columns if column != target]
    if not split_options:
        st.warning("Select a dataset containing a target and at least one feature.")
        return
    if strategy == "group":
        _reset_invalid_selection("ml_group_col", split_options)
        preferred = [column for column in split_options if metadata.get(column, {}).get("role") == "id"]
        group_col = st.selectbox(
            "Group identity column", split_options,
            index=split_options.index(preferred[0]) if preferred else 0, key="ml_group_col",
        )
        st.caption("A group stays entirely in training or test. Test size is a fraction of groups; row counts may differ.")
    elif strategy == "chronological":
        _reset_invalid_selection("ml_time_col", split_options)
        preferred = [
            column for column in split_options
            if pd.api.types.is_datetime64_any_dtype(df[column])
            or metadata.get(column, {}).get("role") == "datetime"
        ]
        time_col = st.selectbox(
            "Time column", split_options,
            index=split_options.index(preferred[0]) if preferred else 0, key="ml_time_col",
        )
        st.caption(
            "Earlier times train; later times test. Equal timestamps stay together. "
            "Use datetime, numeric time, or ISO dates; naive datetimes are interpreted as UTC."
        )

    auxiliary = group_col if strategy == "group" else time_col
    feature_pool = [column for column in eligible if column != target]
    feature_cols = [column for column in feature_pool if column != auxiliary]
    if not feature_cols:
        st.warning("Choose at least one feature besides the target and split column.")
        return
    intent = st.session_state.get("ml_feature_intent", feature_pool)
    scope = (context, auxiliary)
    if st.session_state.get("ml_feature_scope") != scope:
        st.session_state.pop("ml_features", None)
        st.session_state["ml_feature_scope"] = scope
    selected_features = st.multiselect(
        "Features to include", feature_cols,
        default=[column for column in feature_cols if column in intent], key="ml_features",
    )
    st.session_state["ml_feature_intent"] = [
        column for column in feature_pool
        if column in selected_features or (column == auxiliary and column in intent)
    ]
    if not selected_features:
        st.warning("Select at least one feature.")
        return

    with st.expander("Preprocessing preview"):
        categorical = [
            column for column in selected_features
            if not pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        ]
        numeric = [column for column in selected_features if column not in categorical]
        st.write(f"**Categorical** ({len(categorical)}): {categorical}")
        st.write(f"**Numeric** ({len(numeric)}): {numeric}")
        st.caption(
            "Training data determines encodings, feature fills, and scaling. "
            "Categories use at most 64 outputs per column. Missing targets are excluded. "
            "Constant or entirely missing training features are excluded."
        )

    available = get_available_models(problem_type)
    if "ml_models" in st.session_state:
        chosen = st.session_state["ml_models"]
        if any(name not in available for name in chosen):
            st.session_state["ml_models"] = [name for name in chosen if name in available] or list(available)[:2]
    selected_models = st.multiselect(
        "Select models to train", list(available), default=list(available)[:2], key="ml_models",
    )
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="ml_split")
    st.caption(
        "These are results from one holdout split. Interactive feature choices and prior "
        "inspection can affect evaluation; use independent final data for a final performance estimate."
    )

    if not st.button("Train Models", type="primary", key="ml_train"):
        return
    if not selected_models:
        st.warning("Select at least one model.")
        return
    columns = list(dict.fromkeys([*selected_features, target, *([auxiliary] if auxiliary is not None else [])]))
    subset = df.loc[:, columns]
    usable_rows = int(subset[target].notna().sum())
    st.caption(
        f"Using {usable_rows} of {len(subset)} rows; "
        f"excluding {len(subset) - usable_rows} with missing targets."
    )
    with st.spinner("Preprocessing & training..."):
        try:
            X_train, X_test, y_train, y_test, info = auto_preprocess(
                subset, target, metadata, test_size=test_size, problem_type=problem_type,
                split_strategy=strategy, group_col=group_col, time_col=time_col,
            )
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            return
        st.caption(f"Training rows: {info['train_rows']} | Test rows: {info['test_rows']}")
        for message in info["warnings"]:
            st.warning(message)
        results = {}
        for name in selected_models:
            try:
                fitted = train_model(available[name], X_train, y_train)
                metrics = evaluate_model(fitted, X_test, y_test, problem_type)
                metrics["_model"] = fitted
                results[name] = metrics
            except Exception as exc:
                st.error(f"Training **{name}** failed: {exc}")

    if not results:
        return
    st.subheader("Results")
    compare_cols = (
        ["accuracy", "macro_f1", "precision", "recall", "f1"]
        if problem_type == "classification" else ["MAE", "MSE", "RMSE", "R²"]
    )
    comparison = pd.DataFrame([
        {"Model": name, **{column: metrics.get(column) for column in compare_cols}}
        for name, metrics in results.items()
    ]).set_index("Model")
    comparison = comparison.rename(columns={
        "precision": "weighted_precision", "recall": "weighted_recall", "f1": "weighted_f1",
    })
    st.dataframe(comparison)

    for name, metrics in results.items():
        with st.expander(f"Details: {name}"):
            if metrics.get("r2_unavailable_reason"):
                st.info(metrics["r2_unavailable_reason"])
            if problem_type == "classification":
                st.dataframe(pd.DataFrame(metrics["per_class"]).set_index("class"))
            try:
                figure = (
                    plot_confusion_matrix(metrics["confusion_matrix"], metrics["class_labels"])
                    if problem_type == "classification"
                    else plot_residuals(y_test, metrics["y_pred"])
                )
                st.plotly_chart(figure, width="stretch")
            except Exception as exc:
                st.warning(f"Could not render the evaluation plot for {name}: {exc}")
            try:
                importance = plot_feature_importance(metrics["_model"], X_train.columns.tolist())
                if importance is not None:
                    st.plotly_chart(importance, width="stretch")
            except Exception as exc:
                st.warning(f"Could not render feature importance for {name}: {exc}")
