import pandas as pd
import streamlit as st

from backend.ml import (
    detect_problem_type,
    evaluate_model,
    get_available_models,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residuals,
    train_model,
)
from backend.preprocessing import auto_preprocess
from backend.state import StateManager


def data_ml():
    st.title("Machine Learning")

    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return

    df: pd.DataFrame = st.session_state.df
    column_meta: dict = st.session_state.column_meta

    # ── Step 1: Target selection ─────────────────────────────────────────
    eligible = [
        c for c in df.columns
        if column_meta.get(c, {}).get("role") not in ("id", "datetime", "ignore")
    ]
    target = st.selectbox("Select target column", eligible, key="ml_target")
    if not target:
        return

    # ── Step 2: Problem type ─────────────────────────────────────────────
    auto_type = detect_problem_type(df[target])
    problem_type = st.radio(
        "Problem type",
        ["classification", "regression"],
        index=0 if auto_type == "classification" else 1,
        horizontal=True,
        key="ml_ptype",
    )

    # ── Step 3: Feature selection ────────────────────────────────────────
    feature_cols = [c for c in eligible if c != target]
    selected_features = st.multiselect(
        "Features to include",
        feature_cols,
        default=feature_cols,
        key="ml_features",
    )
    if not selected_features:
        st.warning("Select at least one feature.")
        return

    # ── Step 4: Preprocessing preview ────────────────────────────────────
    with st.expander("Preprocessing preview"):
        cat = [c for c in selected_features if df[c].dtype in ("object", "category", "bool")]
        num = [c for c in selected_features if c not in cat]
        st.write(f"**Categorical** ({len(cat)}): {cat}")
        st.write(f"**Numeric** ({len(num)}): {num}")
        st.caption("Categoricals: one-hot if <10 unique values, else label encoding. "
                   "Numerics: standard scaling. NaNs filled with median.")

    # ── Step 5: Model selection ──────────────────────────────────────────
    available = get_available_models(problem_type)
    selected_models = st.multiselect(
        "Select models to train",
        list(available.keys()),
        default=list(available.keys())[:2],
        key="ml_models",
    )

    # ── Step 6: Train/test split ─────────────────────────────────────────
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="ml_split")

    # ── Train ────────────────────────────────────────────────────────────
    if st.button("Train Models", type="primary", key="ml_train"):
        if not selected_models:
            st.warning("Select at least one model.")
            return

        # Subset to selected features + target
        subset = df[selected_features + [target]].dropna()
        if len(subset) < 20:
            st.error("Not enough rows after dropping NaNs. Clean missing values first.")
            return

        with st.spinner("Preprocessing & training..."):
            try:
                X_train, X_test, y_train, y_test, info = auto_preprocess(
                    subset, target, column_meta, test_size=test_size,
                )
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

            results = {}
            for name in selected_models:
                model = available[name]
                try:
                    fitted = train_model(model, X_train, y_train)
                    metrics = evaluate_model(fitted, X_test, y_test, problem_type)
                    metrics["_model"] = fitted
                    results[name] = metrics
                except Exception as e:
                    st.error(f"Training **{name}** failed: {e}")

        if not results:
            return

        # ── Results comparison table ─────────────────────────────────
        st.subheader("Results")

        if problem_type == "classification":
            compare_cols = ["accuracy", "precision", "recall", "f1"]
        else:
            compare_cols = ["MAE", "MSE", "RMSE", "R²"]

        compare_rows = []
        for name, m in results.items():
            row = {"Model": name}
            for c in compare_cols:
                row[c] = m.get(c)
            compare_rows.append(row)
        st.dataframe(pd.DataFrame(compare_rows).set_index("Model"))

        # ── Per-model details ────────────────────────────────────────
        for name, m in results.items():
            with st.expander(f"Details: {name}"):
                fitted = m["_model"]
                y_pred = m["y_pred"]

                if problem_type == "classification":
                    cm = m["confusion_matrix"]
                    fig = plot_confusion_matrix(cm)
                    st.plotly_chart(fig, width='stretch')
                else:
                    fig = plot_residuals(y_test, y_pred)
                    st.plotly_chart(fig, width='stretch')

                fi_fig = plot_feature_importance(fitted, X_train.columns.tolist())
                if fi_fig:
                    st.plotly_chart(fi_fig, width='stretch')
