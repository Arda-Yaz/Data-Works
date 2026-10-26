"""Small, structured diagnostics that exclude uploaded data and error messages."""

from datetime import datetime, timezone
import json
import logging
import math
import re


_LOGGER = logging.getLogger("dataworks")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            **record.dataworks_fields,
        }
        return json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def configure_logging():
    """Configure only the app logger, once; host/Streamlit logging stays intact."""
    if not any(getattr(handler, "dataworks_handler", False) for handler in _LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        handler.dataworks_handler = True
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def log_event(event, **fields):
    """Log a fixed event identifier plus numeric or boolean counters only.

    Callers must use event names from source code, never data, column names,
    filenames, identities, exception messages, or arbitrary user text.
    """
    if not isinstance(event, str) or not _IDENTIFIER.fullmatch(event):
        raise ValueError("A diagnostic event must be a short fixed identifier.")
    safe = {"event": event}
    for key, value in fields.items():
        if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key) or key in ("event", "time", "level"):
            raise ValueError("Diagnostic fields must use short fixed identifiers.")
        if isinstance(value, bool) or isinstance(value, int):
            safe[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            safe[key] = value
        else:
            raise ValueError("Diagnostic fields accept only finite numbers and booleans.")
    _LOGGER.info("", extra={"dataworks_fields": safe})


def log_failure(event, exc):
    """Log only the event and exception class, without message or traceback."""
    if not isinstance(event, str) or not _IDENTIFIER.fullmatch(event):
        raise ValueError("A diagnostic event must be a short fixed identifier.")
    name = type(exc).__name__
    error_type = name if _IDENTIFIER.fullmatch(name) else "Exception"
    _LOGGER.warning("", extra={"dataworks_fields": {"event": event, "error_type": error_type}})
