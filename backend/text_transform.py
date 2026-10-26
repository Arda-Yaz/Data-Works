"""Explicit, lossless-to-missing text edits for arbitrary tabular datasets."""

import copy
import re
import unicodedata

import pandas as pd


_WHITESPACE = re.compile(r"\s+")
_CASES = {"keep", "lower", "upper", "casefold"}
_UNICODE_FORMS = {None, "NFC", "NFKC"}


def validate_normalization(*, strip=False, collapse_whitespace=False, case="keep", unicode_form=None):
    if not isinstance(strip, bool) or not isinstance(collapse_whitespace, bool):
        raise ValueError("Text normalization switches must be booleans.")
    if not isinstance(case, str) or case not in _CASES:
        raise ValueError("Text case must be keep, lower, upper, or casefold.")
    if unicode_form is not None and (not isinstance(unicode_form, str) or unicode_form not in _UNICODE_FORMS):
        raise ValueError("Unicode normalization must be NFC, NFKC, or None.")
    return {"strip": strip, "collapse_whitespace": collapse_whitespace, "case": case, "unicode_form": unicode_form}


def validate_category_mapping(mapping):
    if not isinstance(mapping, dict) or any(
        not isinstance(source, str) or not isinstance(target, str)
        for source, target in mapping.items()
    ):
        raise ValueError("Category mappings must map exact text values to text replacements.")
    return copy.deepcopy(mapping)


def _normalizer(*, strip, collapse_whitespace, case, unicode_form):
    def transform(value):
        if unicode_form is not None:
            value = unicodedata.normalize(unicode_form, value)
        if collapse_whitespace:
            value = _WHITESPACE.sub(" ", value)
        if strip:
            value = value.strip()
        return value if case == "keep" else getattr(value, case)()
    return transform


def _text_series(series):
    dtype = series.dtype
    if isinstance(dtype, pd.StringDtype):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        return any(isinstance(value, str) for value in series.cat.categories) or len(series.cat.categories) == 0
    if pd.api.types.is_object_dtype(dtype):
        return any(isinstance(value, str) for value in series) or series.isna().all()
    return False


def get_text_columns(df):
    """Return columns with text storage/content, including mixed object columns."""
    _validate_frame(df)
    return [column for column in df.columns if _text_series(df[column])]


def _validate_frame(df):
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("Text transformations require a DataFrame with unique columns.")


def _validate_columns(df, columns):
    _validate_frame(df)
    if not isinstance(columns, list) or not columns:
        raise ValueError("Choose at least one text column.")
    try:
        if len(set(columns)) != len(columns):
            raise ValueError("Choose each text column once.")
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column {column!r} is not present in the dataset.")
            if not _text_series(df[column]):
                raise ValueError(f"Column {column!r} has no text values; choose a text column.")
    except TypeError as exc:
        raise ValueError("Text column references must be valid column labels.") from exc


def _transform_series(series, transform):
    def apply(value):
        return transform(value) if isinstance(value, str) else value

    values = [apply(value) for value in series.array]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = list(dict.fromkeys(apply(value) for value in series.cat.categories))
        array = pd.Categorical(values, categories=categories, ordered=series.cat.ordered)
        return pd.Series(array, index=series.index, name=series.name)
    if isinstance(series.dtype, pd.StringDtype):
        return pd.Series(pd.array(values, dtype=series.dtype), index=series.index, name=series.name)
    return pd.Series(values, index=series.index, name=series.name, dtype=object)


def normalize_text(df, columns, *, strip=False, collapse_whitespace=False, case="keep", unicode_form=None):
    """Edit strings only; preserve missing and non-string values in mixed columns.

    Whitespace collapse retains one edge space unless strip=True. Category levels
    (including unused levels) are transformed in their original order; collisions
    merge into the first corresponding category. Input data is never modified.
    """
    options = validate_normalization(strip=strip, collapse_whitespace=collapse_whitespace, case=case, unicode_form=unicode_form)
    _validate_columns(df, columns)
    transform = _normalizer(**options)
    result = df.copy(deep=True)
    for column in columns:
        result[column] = _transform_series(df[column], transform)
    return result


def map_categories(df, column, mapping):
    """Replace exact text values simultaneously, retaining unmatched values."""
    mapping = validate_category_mapping(mapping)
    _validate_columns(df, [column])
    result = df.copy(deep=True)
    result[column] = _transform_series(df[column], lambda value: mapping.get(value, value))
    return result


def transformed_metadata(metadata, columns, *, normalization=None, mapping=None):
    """Apply the same explicit text edits to allowed-value rules."""
    result = copy.deepcopy(metadata)
    transform = _normalizer(**validate_normalization(**normalization)) if normalization is not None else None
    if mapping is not None:
        owned_mapping = validate_category_mapping(mapping)
        transform = lambda value: owned_mapping.get(value, value)
    if transform is None:
        raise ValueError("Choose a text transformation for the metadata.")
    for column in columns:
        allowed = result.get(column, {}).get("allowed_values")
        if allowed is not None:
            if not isinstance(allowed, list) or any(not isinstance(value, str) for value in allowed):
                raise ValueError(f"Column {column!r}: allowed values must be a list of text.")
            result[column]["allowed_values"] = list(dict.fromkeys(transform(value) for value in allowed))
    return result
