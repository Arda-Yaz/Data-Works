from io import StringIO
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from backend.access import require_access
from backend import diagnostics


class StopScript(BaseException):
    pass


class User(dict):
    def __init__(self, logged_in=False, **claims):
        super().__init__(claims)
        self.is_logged_in = logged_in


AUTH = {
    "redirect_uri": "https://app.example.com/oauth2callback",
    "cookie_secret": "x" * 48,
    "client_id": "client",
    "client_secret": "secret",
    "server_metadata_url": "https://identity.example.com/.well-known/openid-configuration",
}


class RuntimeReadinessTests(unittest.TestCase):
    def test_diagnostics_are_json_and_exclude_exception_messages(self):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(diagnostics._JsonFormatter())
        logger = diagnostics._LOGGER
        previous_handlers, previous_level, previous_propagate = logger.handlers, logger.level, logger.propagate
        try:
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)
            logger.propagate = False
            diagnostics.log_event("dataset_loaded", rows=12, columns=4, limited=False)
            diagnostics.log_failure("dataset_failed", ValueError("SECRET cell, user@example.com, C:/private.csv"))
        finally:
            logger.handlers, logger.level, logger.propagate = previous_handlers, previous_level, previous_propagate
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(records[0]["rows"], 12)
        self.assertEqual(records[0]["columns"], 4)
        self.assertEqual(records[1]["error_type"], "ValueError")
        self.assertNotIn("SECRET", stream.getvalue())
        self.assertNotIn("user@example.com", stream.getvalue())
        self.assertNotIn("private.csv", stream.getvalue())
        self.assertNotIn("traceback", stream.getvalue().lower())

    def test_diagnostics_reject_arbitrary_user_fields(self):
        for fields in ({"filename": "private.csv"}, {"rows": float("nan")}, {"time": 1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                diagnostics.log_event("operation_failed", **fields)
        with self.assertRaises(ValueError):
            diagnostics.log_event("private filename.csv")

    def test_logging_configuration_is_idempotent_and_does_not_change_root_logger(self):
        logger = diagnostics._LOGGER
        previous_handlers, previous_level, previous_propagate = logger.handlers, logger.level, logger.propagate
        root_handlers = list(logging.getLogger().handlers)
        try:
            logger.handlers = []
            diagnostics.configure_logging()
            diagnostics.configure_logging()
            self.assertEqual(len(logger.handlers), 1)
            self.assertEqual(logging.getLogger().handlers, root_handlers)
            self.assertFalse(logger.propagate)
        finally:
            for handler in logger.handlers:
                if handler not in previous_handlers:
                    handler.close()
            logger.handlers, logger.level, logger.propagate = previous_handlers, previous_level, previous_propagate

    def ui(self, user=None, auth=None):
        fake = MagicMock()
        fake.secrets = {"auth": dict(AUTH if auth is None else auth)}
        fake.user = user or User()
        fake.stop.side_effect = StopScript
        fake.button.return_value = False
        fake.sidebar.button.return_value = False
        return fake

    def test_auth_disabled_does_not_read_secrets_or_identity(self):
        with patch.dict("os.environ", {}, clear=True), patch("backend.access.st") as fake:
            self.assertTrue(require_access())
            fake.stop.assert_not_called()
            fake.login.assert_not_called()
            self.assertEqual(fake.mock_calls, [])

    def test_invalid_auth_switch_stops_before_dataset_pages(self):
        fake = self.ui()
        with patch.dict("os.environ", {"DATAWORKS_REQUIRE_AUTH": "typo"}, clear=True), patch("backend.access.st", fake):
            with self.assertRaises(StopScript):
                require_access()
        fake.error.assert_called_once()
        fake.login.assert_not_called()

    def test_required_but_incomplete_or_insecure_configuration_fails_closed(self):
        configurations = [{}, {**AUTH, "cookie_secret": "short"}, {**AUTH, "redirect_uri": "http://public.example.com/oauth2callback"}, {**AUTH, "server_metadata_url": "file:///private"}]
        for auth in configurations:
            fake = self.ui(User(True), auth=auth)
            with self.subTest(auth=auth), patch.dict("os.environ", {"DATAWORKS_REQUIRE_AUTH": "true"}, clear=True), patch("backend.access.st", fake), patch("backend.access.metadata.version", return_value="1.9.1"):
                with self.assertRaises(StopScript):
                    require_access()
            fake.error.assert_called_once()
            fake.login.assert_not_called()

    def test_unauthenticated_user_stops_and_uses_native_login_only_on_click(self):
        fake = self.ui()
        with patch.dict("os.environ", {"DATAWORKS_REQUIRE_AUTH": "true"}, clear=True), patch("backend.access.st", fake), patch("backend.access.metadata.version", return_value="1.9.1"):
            with self.assertRaises(StopScript):
                require_access()
            fake.login.assert_not_called()
            fake.button.return_value = True
            with self.assertRaises(StopScript):
                require_access()
        fake.login.assert_called_once_with()

    def test_allowlist_requires_exact_verified_email(self):
        permitted = User(True, email="ANALYST@example.com", email_verified=True)
        denied = [User(True, email="analyst@example.com.attacker.test", email_verified=True), User(True, email="analyst@example.com"), User(True, email="analyst@example.com", email_verified="true")]
        environment = {"DATAWORKS_REQUIRE_AUTH": "true", "DATAWORKS_ALLOWED_EMAILS": "analyst@example.com"}
        with patch.dict("os.environ", environment, clear=True), patch("backend.access.metadata.version", return_value="1.9.1"):
            fake = self.ui(permitted)
            with patch("backend.access.st", fake):
                self.assertTrue(require_access())
            for user in denied:
                fake = self.ui(user)
                with self.subTest(user=user), patch("backend.access.st", fake):
                    with self.assertRaises(StopScript):
                        require_access()
                fake.error.assert_called_once()

    def test_signed_in_user_can_sign_out_without_accessing_pages(self):
        fake = self.ui(User(True, sub="account"))
        fake.sidebar.button.return_value = True
        with patch.dict("os.environ", {"DATAWORKS_REQUIRE_AUTH": "true"}, clear=True), patch("backend.access.st", fake), patch("backend.access.metadata.version", return_value="1.9.1"):
            with self.assertRaises(StopScript):
                require_access()
        fake.logout.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
