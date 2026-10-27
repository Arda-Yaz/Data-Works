"""Prevent old project readers from silently discarding fulfillment evidence."""

import io
import json
from pathlib import Path
import unittest
import zipfile

import pandas as pd
from pandas.testing import assert_frame_equal

from backend.fulfillment import ingest_csv
from backend.fulfillment_state import new_project, run_project, set_source, validate_project
from backend.project_io import (
    FORMAT, FORMAT_VERSION, FULFILLMENT_FORMAT_VERSION, dumps_project, encode_value, loads_project,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"


def envelope(archive_bytes):
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return json.loads(archive.read("project.json"))


def archive_for(payload, version):
    output = io.BytesIO()
    data = {"format": FORMAT, "version": version, "payload": encode_value(payload)}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(data))
    return output.getvalue()


class FulfillmentArchiveVersionTests(unittest.TestCase):
    def completed(self):
        source = ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        return run_project(set_source(new_project(), source))

    def test_generic_projects_retain_legacy_version_one_and_round_trip(self):
        self.assertEqual(FORMAT_VERSION, 1)
        payload = {"df": pd.DataFrame({"identifier": ["0001", "0002"]}), "name": "generic"}
        archived = dumps_project(payload)
        self.assertEqual(envelope(archived)["version"], 1)
        reopened = loads_project(archived)
        self.assertEqual(reopened["name"], payload["name"])
        assert_frame_equal(reopened["df"], payload["df"])

    def test_fulfillment_only_projects_write_v2_and_preserve_completed_evidence(self):
        self.assertEqual(FULFILLMENT_FORMAT_VERSION, 2)
        completed = self.completed()
        payload = {"mode": "fulfillment-only", "project_name": "Orders", "fulfillment": completed}
        archived = dumps_project(payload)
        header = envelope(archived)
        self.assertEqual(header["version"], 2)
        # Actual legacy readers require exactly version 1 before decoding the
        # payload. The new envelope intentionally fails that existing guard.
        legacy_version_supported = type(header["version"]) is int and header["version"] == 1
        self.assertFalse(legacy_version_supported)
        reopened = loads_project(archived)
        self.assertEqual(reopened["mode"], "fulfillment-only")
        self.assertEqual(validate_project(reopened["fulfillment"]), completed)

    def test_mixed_projects_write_v2_instead_of_risking_legacy_ledger_loss(self):
        completed = self.completed()
        payload = {"df": pd.DataFrame({"unrelated": [1, 2]}), "fulfillment": completed}
        archived = dumps_project(payload)
        self.assertEqual(envelope(archived)["version"], 2)
        reopened = loads_project(archived)
        assert_frame_equal(reopened["df"], payload["df"])
        self.assertEqual(validate_project(reopened["fulfillment"]), completed)

    def test_fulfillment_payloads_cannot_be_downgraded_to_v1(self):
        for payload in ({"fulfillment": self.completed()}, {"fulfillment": None}):
            with self.subTest(valid_extension=payload["fulfillment"] is not None), self.assertRaisesRegex(ValueError, "fulfillment"):
                loads_project(archive_for(payload, 1))

    def test_v2_requires_a_top_level_fulfillment_payload(self):
        for payload in ({}, {"df": pd.DataFrame({"x": [1]})}, {"nested": {"fulfillment": {}}}):
            with self.subTest(fields=list(payload)), self.assertRaisesRegex(ValueError, "fulfillment"):
                loads_project(archive_for(payload, 2))

    def test_unknown_or_noninteger_versions_fail_before_payload_restoration(self):
        for version in (0, 3, 999, True, False, 1.0, 2.0, "2", None):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "version"):
                loads_project(archive_for({"fulfillment": self.completed()}, version))

    def test_only_the_reserved_top_level_field_selects_v2(self):
        payload = {"notes": {"fulfillment": "ordinary nested metadata"}}
        archived = dumps_project(payload)
        self.assertEqual(envelope(archived)["version"], 1)
        self.assertEqual(loads_project(archived), payload)

    def test_codec_version_check_does_not_replace_extension_semantic_validation(self):
        archived = dumps_project({"fulfillment": None})
        self.assertEqual(envelope(archived)["version"], 2)
        reopened = loads_project(archived)
        self.assertIsNone(reopened["fulfillment"])
        with self.assertRaises(ValueError):
            validate_project(reopened["fulfillment"])


if __name__ == "__main__":
    unittest.main()
