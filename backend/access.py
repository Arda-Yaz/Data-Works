"""Optional native Streamlit OIDC gate, evaluated before dataset page access."""

from collections.abc import Mapping
from importlib import metadata
import os
from urllib.parse import urlparse

import streamlit as st

from backend.diagnostics import log_event, log_failure


def _stop_error(message):
    st.error(message)
    st.stop()
    return False


def _authentication_enabled():
    value = os.environ.get("DATAWORKS_REQUIRE_AUTH", "false").strip().casefold()
    if value in ("1", "true", "yes"):
        return True
    if value in ("", "0", "false", "no"):
        return False
    raise ValueError("DATAWORKS_REQUIRE_AUTH must be true or false.")


def _allowed_emails():
    value = os.environ.get("DATAWORKS_ALLOWED_EMAILS", "").strip()
    if not value:
        return set()
    emails = {item.strip().casefold() for item in value.split(",")}
    if any(not item or item.count("@") != 1 or any(character.isspace() for character in item) for item in emails):
        raise ValueError("DATAWORKS_ALLOWED_EMAILS must contain comma-separated full email addresses.")
    return emails


def _validate_authentication():
    try:
        installed = metadata.version("Authlib")
        release = tuple(int(part) for part in installed.split(".")[:3])
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise ValueError("Install the pinned requirements before enabling authentication.") from exc
    if release < (1, 3, 2):
        raise ValueError("The installed authentication dependency is too old.")
    auth = st.secrets.get("auth", {})
    required = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
    if not isinstance(auth, Mapping) or any(not isinstance(auth.get(key), str) or not auth[key].strip() for key in required):
        raise ValueError("Configure the default OIDC provider in Streamlit secrets before enabling authentication.")
    if len(auth["cookie_secret"]) < 32:
        raise ValueError("The authentication cookie secret must contain at least 32 characters.")
    for key in ("redirect_uri", "server_metadata_url"):
        parsed = urlparse(auth[key])
        local_http = parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if not parsed.hostname or not (parsed.scheme == "https" or local_http) or parsed.username or parsed.password:
            raise ValueError("OIDC URLs must use HTTPS, except local development URLs.")
    if not urlparse(auth["redirect_uri"]).path.endswith("/oauth2callback"):
        raise ValueError("The authentication redirect URI must end in /oauth2callback.")


def require_access():
    """Return True for permitted sessions; stop the script for all other cases."""
    try:
        enabled = _authentication_enabled()
    except ValueError as exc:
        log_failure("auth_configuration_failed", exc)
        return _stop_error("Authentication settings are invalid. Ask the operator to check the deployment configuration.")
    if not enabled:
        return True
    try:
        _validate_authentication()
        allowed = _allowed_emails()
        logged_in = bool(st.user.is_logged_in)
    except Exception as exc:
        log_failure("auth_configuration_failed", exc)
        return _stop_error("Authentication is required but is not configured correctly. Ask the operator to check the deployment configuration.")
    if not logged_in:
        st.info("Sign in to access DataWorks.")
        if st.button("Sign in", key="auth_sign_in"):
            try:
                st.login()
            except Exception as exc:
                log_failure("auth_login_failed", exc)
                st.error("Sign-in could not start. Ask the operator to check the identity-provider configuration.")
        st.stop()
        return False
    if allowed:
        email = st.user.get("email")
        verified = st.user.get("email_verified") is True
        if not verified or not isinstance(email, str) or email.casefold() not in allowed:
            log_event("auth_access_denied")
            st.error("This account is not permitted to access DataWorks.")
            if st.button("Sign out", key="auth_denied_sign_out"):
                st.logout()
            st.stop()
            return False
    if st.sidebar.button("Sign out", key="auth_sign_out"):
        st.logout()
        st.stop()
        return False
    return True
