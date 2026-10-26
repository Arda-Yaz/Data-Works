import pandas as pd
import streamlit as st

from backend.ingestion import apply_inferred_dtypes
from backend.experiments import (
    MAX_EXPERIMENTS, create_experiment, experiment_comparison, export_experiment_csv,
    export_experiment_json, export_model_artifact, importance_figure,
)
from backend.ml import (
    detect_problem_type, evaluate_model, get_available_models,
    plot_confusion_matrix, plot_feature_importance, plot_residuals, train_model,
)
from backend.preprocessing import auto_preprocess
from backend.state import StateManager
from backend.resources import ml_job
from backend.diagnostics import log_event, log_failure


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


def _training_controls():
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
    try:
        with st.spinner("Preprocessing & training..."), ml_job():
            try:
                X_train, X_test, y_train, y_test, info = auto_preprocess(
                    subset, target, metadata, test_size=test_size, problem_type=problem_type,
                    split_strategy=strategy, group_col=group_col, time_col=time_col,
                )
            except Exception as exc:
                log_failure("ml_preprocessing_failed", exc)
                st.error(f"Preprocessing failed: {exc}")
                return
            st.caption(f"Training rows: {info['train_rows']} | Test rows: {info['test_rows']}")
            for message in info["warnings"]:
                st.warning(message)
            results, errors = {}, {}
            for name in selected_models:
                try:
                    fitted = train_model(available[name], X_train, y_train)
                    metrics = evaluate_model(fitted, X_test, y_test, problem_type)
                    metrics["_model"] = fitted
                    results[name] = metrics
                    log_event("ml_model_completed", training_rows=len(X_train), test_rows=len(X_test), features=X_train.shape[1])
                except Exception as exc:
                    log_failure("ml_model_failed", exc)
                    errors[name] = str(exc)
                    st.error(f"Training **{name}** failed: {exc}")
    except ValueError as exc:
        log_failure("ml_run_unavailable", exc)
        st.error(str(exc))
        return

    if not results:
        return
    config = {
        "source": source, "target": target, "features": list(selected_features),
        "problem_type": problem_type, "split_strategy": strategy, "test_size": test_size,
        "group_col": group_col, "time_col": time_col, "random_state": 42,
        "selected_models": list(selected_models),
    }
    try:
        experiment = create_experiment(
            dataset_id=st.session_state.dataset_id, data_version=st.session_state.data_version,
            config=config, results=results, errors=errors, y_test=y_test,
            feature_names=X_train.columns.tolist(),
            feature_columns=info.get("feature_columns") or getattr(info["preprocessor"], "source_columns_", selected_features),
            info=info, schema=metadata,
        )
        identifier = StateManager.add_experiment(experiment)
    except Exception as exc:
        log_failure("ml_report_retention_failed", exc)
        st.error(f"This run could not be retained. {exc}")
        return
    st.session_state["ml_experiment_id"] = identifier


def _render_experiments():
    experiments = st.session_state.experiments
    if not experiments:
        st.selectbox(
            "Completed experiment", [], index=None, disabled=True,
            placeholder="Train a model to save an experiment.", key="ml_experiment_id",
        )
        return
    st.subheader("Saved Experiments")
    st.caption(f"The latest {MAX_EXPERIMENTS} completed runs are retained for this dataset. Download reports to keep more runs.")
    ordered = list(reversed(experiments))
    identifiers = [experiment["id"] for experiment in ordered]
    by_id = {experiment["id"]: experiment for experiment in ordered}
    _reset_invalid_selection("ml_experiment_id", identifiers)

    def label(identifier):
        record = by_id[identifier]
        return f"{record['created_at']} | {record['config']['target']} | {record['config']['problem_type']}"

    selected = st.selectbox("Completed experiment", identifiers, format_func=label, key="ml_experiment_id")
    experiment = by_id[selected]
    config = experiment["config"]
    st.caption(
        f"Stored run {experiment['id'][:8]} · Created UTC: {experiment['created_at']} · "
        f"Target: {config['target']} · Source: {_SOURCE_LABELS.get(config['source'], config['source'])} · "
        f"Split: {_SPLIT_LABELS.get(config['split_strategy'], config['split_strategy'])} · "
        f"Test fraction: {config['test_size']} · Features: {config['features']}"
    )
    if experiment["data_version"] != st.session_state.data_version:
        st.warning("This run belongs to an earlier dataset state. Its scores describe that stored run; retrain to evaluate current data.")
    else:
        st.caption("This run uses the current dataset state. Its stored settings may differ from the controls above.")
    for name, message in experiment.get("errors", {}).items():
        st.warning(f"This run did not complete {name}: {message}")
    for message in experiment.get("preprocessing_info", {}).get("warnings", []):
        st.caption(message)

    first, second, third = st.columns(3)
    with first:
        st.download_button(
            "Download experiment report (JSON)", lambda: export_experiment_json(experiment),
            file_name=f"dataworks-experiment-{experiment['id'][:8]}.json", mime="application/json",
            key="ml_report_json", on_click="ignore",
        )
    with second:
        st.download_button(
            "Download model comparison (CSV)", lambda: export_experiment_csv(experiment),
            file_name=f"dataworks-comparison-{experiment['id'][:8]}.csv", mime="text/csv",
            key="ml_report_csv", on_click="ignore",
        )
    with third:
        if st.button("Delete this experiment", key="ml_delete_experiment"):
            StateManager.remove_experiment(selected)
            st.rerun()

    st.subheader("Results")
    st.dataframe(experiment_comparison(experiment))
    runtime = experiment.get("preprocessor") is not None and bool(experiment.get("models"))
    if runtime:
        st.caption(
            "Python model downloads include fitted preprocessing and need a compatible DataWorks installation. "
            "Joblib files can execute code when loaded; load only trusted exports. This app does not import model binaries."
        )
    else:
        st.info("This reopened report contains no live fitted models. Retrain to download a Python model artifact.")

    for position, (name, metrics) in enumerate(experiment["results"].items()):
        with st.expander(f"Details: {name}"):
            if metrics.get("r2_unavailable_reason"):
                st.info(metrics["r2_unavailable_reason"])
            if config["problem_type"] == "classification":
                st.dataframe(pd.DataFrame(metrics["per_class"]).set_index("class"))
            try:
                figure = (
                    plot_confusion_matrix(metrics["confusion_matrix"], metrics["class_labels"])
                    if config["problem_type"] == "classification"
                    else plot_residuals(experiment["y_test"], metrics["y_pred"])
                )
                st.plotly_chart(figure, width="stretch", key=f"ml_evaluation_{experiment['id']}_{position}")
            except Exception as exc:
                st.warning(f"Could not render the stored evaluation plot for {name}: {exc}")
            try:
                figure = importance_figure(experiment, name)
                if figure is not None:
                    st.plotly_chart(figure, width="stretch", key=f"ml_importance_{experiment['id']}_{position}")
                elif name in experiment.get("plot_errors", {}):
                    st.warning(f"Could not render feature importance for {name}: {experiment['plot_errors'][name]}")
            except Exception as exc:
                st.warning(f"Could not render stored feature importance for {name}: {exc}")
            if runtime and name in experiment.get("models", {}):
                st.download_button(
                    "Download Python model (.joblib)",
                    lambda record=experiment, model=name: export_model_artifact(record, model),
                    file_name=f"dataworks-model-{experiment['id'][:8]}-{position}.joblib",
                    mime="application/octet-stream", key=f"ml_model_export_{experiment['id']}_{position}",
                    on_click="ignore",
                )


def data_ml():
    st.title("Machine Learning")
    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return
    _training_controls()
    _render_experiments()
