"""Compact persistence must retain evidence, ownership, and compatibility gates."""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from backend import resources
from backend.fulfillment import ingest_csv
from backend.fulfillment_state import new_project, run_project, set_source, validate_project
from backend.fulfillment_wire import FORMAT as WIRE_FORMAT, pack_fulfillment, unpack_fulfillment
from backend.project_io import (
    FORMAT, FULFILLMENT_FORMAT_VERSION, MAX_ARCHIVE_BYTES, MAX_JSON_BYTES, MAX_NODES,
    dumps_project, encode_value, loads_project,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"


def archive_for(payload):
    output = io.BytesIO()
    document = {"format": FORMAT, "version": FULFILLMENT_FORMAT_VERSION,
                "payload": encode_value(payload)}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(document))
    return output.getvalue()


class FulfillmentWireTests(unittest.TestCase):
    def completed(self):
        source = ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        return run_project(set_source(new_project(), source))

    def payload(self, project):
        return {"mode": "fulfillment-only", "project_name": "Orders", "fulfillment": project}

    def test_wire_stores_original_sources_and_actual_attempts_once(self):
        project = self.completed()
        before = copy.deepcopy(project)
        wire = pack_fulfillment(project)
        self.assertEqual(project, before)
        self.assertEqual(wire["format"], WIRE_FORMAT)
        self.assertEqual(type(wire["version"]), int)
        self.assertEqual(wire["version"], 2)
        self.assertEqual(len(wire["sources"]), 1)
        self.assertEqual(len(wire["plans"]), 1)
        self.assertEqual(len(wire["attempts"]), 1)
        self.assertEqual(wire["current"]["last_attempt"], wire["current"]["committed"]["attempt"])
        stored_source = next(iter(wire["sources"].values()))
        self.assertEqual(set(stored_source), {"raw_bytes", "read_options"})
        self.assertEqual(stored_source["raw_bytes"], project["source"]["raw_bytes"])
        self.assertNotIn("rows", stored_source)
        stored_attempt = next(iter(wire["attempts"].values()))
        self.assertEqual(stored_attempt, project["last_attempt"])
        self.assertEqual(stored_attempt["rejected"], project["committed"]["attempt"]["rejected"])
        self.assertEqual(unpack_fulfillment(wire), before)

    def test_codec_writes_compact_wire_and_returns_the_unchanged_public_shape(self):
        project = self.completed()
        payload = self.payload(project)
        data = dumps_project(payload)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            document = json.loads(archive.read("project.json"))
        self.assertEqual(document["version"], 2)
        self.assertEqual(document["payload"]["fulfillment"]["format"], WIRE_FORMAT)
        self.assertEqual(document["payload"]["fulfillment"]["version"], 2)
        reopened = loads_project(data)
        self.assertEqual(reopened, payload)
        self.assertEqual(reopened["fulfillment"]["version"], 1)
        self.assertNotIn("format", reopened["fulfillment"])
        self.assertEqual(validate_project(reopened["fulfillment"]), project)

    def test_inflated_mutable_occurrences_are_independent_across_current_and_history(self):
        project = self.completed()
        checkpoint = {key: copy.deepcopy(project[key]) for key in ("source", "draft_plan", "last_attempt", "committed")}
        project["history"].append(checkpoint)
        original = copy.deepcopy(project)
        wire = pack_fulfillment(project)
        restored = unpack_fulfillment(wire)
        self.assertEqual(restored, original)
        restored["source"]["rows"][0][0] = "changed source"
        restored["last_attempt"]["accepted"][0]["values"][0] = "changed attempt"
        restored["draft_plan"]["steps"][1]["status_mapping"]["pending"] = "fulfilled"
        self.assertEqual(restored["committed"]["source"]["rows"][0][0], "000101")
        self.assertEqual(restored["committed"]["attempt"]["accepted"][0]["values"][0], "000101")
        self.assertEqual(restored["committed"]["plan"]["steps"][1]["status_mapping"]["pending"], "pending")
        self.assertEqual(restored["history"][-1], checkpoint)
        self.assertEqual(project, original)
        self.assertEqual(next(iter(wire["attempts"].values())), original["last_attempt"])

    def test_repeated_source_identifier_cannot_hide_different_parsed_evidence(self):
        project = self.completed()
        project["committed"]["source"]["rows"][0][0] = "tampered"
        with self.assertRaisesRegex(ValueError, "source"):
            pack_fulfillment(project)

    def test_repeated_run_identifier_cannot_hide_different_typed_evidence(self):
        for value in (4.0, True, 99):
            project = self.completed()
            project["committed"]["attempt"]["report"]["accepted_rows"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "inconsistent"):
                pack_fulfillment(project)

    def test_precompact_fulfillment_archives_remain_readable(self):
        payload = self.payload(self.completed())
        old_archive = archive_for(payload)
        reopened = loads_project(old_archive)
        self.assertEqual(reopened, payload)
        self.assertEqual(validate_project(reopened["fulfillment"]), payload["fulfillment"])

    def test_unknown_wire_format_versions_and_top_level_fields_fail_explicitly(self):
        wire = pack_fulfillment(self.completed())
        changes = [
            lambda item: item.update(format="foreign-fulfillment-wire"),
            lambda item: item.update(version=1),
            lambda item: item.update(version=3),
            lambda item: item.update(version=True),
            lambda item: item.update(version=2.0),
            lambda item: item.update(arbitrary_python="never execute"),
            lambda item: item.pop("format"),
        ]
        for index, change in enumerate(changes):
            invalid = copy.deepcopy(wire)
            change(invalid)
            with self.subTest(change=index), self.assertRaises(ValueError):
                loads_project(archive_for(self.payload(invalid)))

    def test_dangling_source_plan_and_attempt_references_are_rejected(self):
        wire = pack_fulfillment(self.completed())
        for field, invalid_reference in (("source", "f" * 64), ("draft_plan", "f" * 64),
                                          ("last_attempt", "f" * 32)):
            invalid = copy.deepcopy(wire)
            invalid["current"][field] = invalid_reference
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "dangling"):
                unpack_fulfillment(invalid)

    def test_unused_table_entries_are_rejected_before_source_parsing(self):
        wire = pack_fulfillment(self.completed())
        for name in ("sources", "plans", "attempts"):
            invalid = copy.deepcopy(wire)
            invalid[name]["f" * (32 if name == "attempts" else 64)] = copy.deepcopy(next(iter(invalid[name].values())))
            with self.subTest(table=name), patch("backend.fulfillment_wire.ingest_csv", side_effect=AssertionError("must not parse unused data")):
                with self.assertRaisesRegex(ValueError, "unused"):
                    unpack_fulfillment(invalid)

    def test_table_identifiers_must_match_exact_source_plan_and_run_evidence(self):
        wire = pack_fulfillment(self.completed())
        source_changed = copy.deepcopy(wire)
        next(iter(source_changed["sources"].values()))["read_options"]["encoding"] = "utf-8"
        plan_changed = copy.deepcopy(wire)
        next(iter(plan_changed["plans"].values()))["revision"] += 1
        attempt_changed = copy.deepcopy(wire)
        next(iter(attempt_changed["attempts"].values()))["run_id"] = "f" * 32
        for index, invalid in enumerate((source_changed, plan_changed, attempt_changed)):
            with self.subTest(table=index), self.assertRaisesRegex(ValueError, "table key"):
                unpack_fulfillment(invalid)

    def test_attempt_references_cannot_be_rebound_to_another_source_or_plan(self):
        project = self.completed()
        source_b = ingest_csv((FIXTURES / "month_b.csv").read_bytes())
        project = set_source(project, source_b)
        # The old committed run remains available; claiming it as a new-file
        # attempt must not reattribute the earlier result.
        wire = pack_fulfillment(project)
        invalid = copy.deepcopy(wire)
        invalid["current"]["last_attempt"] = invalid["current"]["committed"]["attempt"]
        with self.assertRaisesRegex(ValueError, "source or plan"):
            unpack_fulfillment(invalid)

    def test_actual_invalid_output_is_preserved_for_semantic_rejection_not_regenerated(self):
        project = self.completed()
        for attempt in (project["last_attempt"], project["committed"]["attempt"]):
            attempt["accepted"][0]["values"][0] = "forged-output"
        loaded = loads_project(dumps_project(self.payload(project)))
        self.assertEqual(loaded["fulfillment"]["last_attempt"]["accepted"][0]["values"][0], "forged-output")
        with self.assertRaises(ValueError):
            validate_project(loaded["fulfillment"])

    def test_history_count_and_inflated_byte_budgets_apply_before_public_restoration(self):
        project = self.completed()
        wire = pack_fulfillment(project)
        excessive = copy.deepcopy(wire)
        excessive["history"] = [copy.deepcopy(wire["current"])] * 11
        with self.assertRaisesRegex(ValueError, "undo"):
            unpack_fulfillment(excessive)
        with patch.object(resources, "MAX_HISTORY_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "memory budget"):
                pack_fulfillment(project)
            with self.assertRaisesRegex(ValueError, "memory budget"):
                unpack_fulfillment(wire)

    def test_table_count_cannot_exceed_checkpoint_reference_budget(self):
        wire = pack_fulfillment(self.completed())
        for name in ("sources", "plans", "attempts"):
            invalid = copy.deepcopy(wire)
            template = next(iter(invalid[name].values()))
            width = 32 if name == "attempts" else 64
            invalid[name] = {format(index, f"0{width}x"): template for index in range(23)}
            with self.subTest(table=name), self.assertRaisesRegex(ValueError, "reference budget"):
                unpack_fulfillment(invalid)

    def test_no_history_and_empty_runtime_states_keep_their_exact_meaning(self):
        empty = new_project()
        self.assertEqual(unpack_fulfillment(pack_fulfillment(empty)), empty)
        project = self.completed()
        project["history"] = []
        loaded = loads_project(dumps_project(self.payload(project)))
        self.assertEqual(loaded["fulfillment"]["history"], [])
        self.assertEqual(loaded["fulfillment"], project)
        self.assertEqual(MAX_ARCHIVE_BYTES, 50 * 1024 * 1024)
        self.assertEqual(MAX_JSON_BYTES, 100 * 1024 * 1024)
        self.assertEqual(MAX_NODES, 10_000_000)


if __name__ == "__main__":
    unittest.main()
