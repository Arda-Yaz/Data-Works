"""Executable plan/source contracts; no widget state or silent fallback allowed."""

import copy
import csv
import importlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fulfillment_v1"


class FulfillmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = importlib.import_module("backend.fulfillment")

    def test_default_plan_is_versioned_json_data_with_a_stable_owned_roundtrip(self):
        plan = self.engine.default_plan()
        self.assertEqual(plan["format"], "dataworks-fulfillment-plan")
        self.assertEqual(type(plan["version"]), int)
        self.assertEqual(plan["version"], 1)
        self.assertEqual(type(plan["revision"]), int)
        self.assertGreaterEqual(plan["revision"], 1)
        self.assertRegex(plan["plan_id"], r"^[0-9a-f]{32}$")
        serialized = self.engine.dumps_plan(plan)
        self.assertIsInstance(serialized, bytes)
        self.assertIsInstance(json.loads(serialized), dict)
        reopened = self.engine.loads_plan(serialized)
        self.assertEqual(reopened, plan)
        self.assertIsNot(reopened, plan)
        self.assertEqual(self.engine.plan_digest(reopened), self.engine.plan_digest(plan))
        self.assertRegex(self.engine.plan_digest(plan), r"^[0-9a-f]{64}$")
        reopened["revision"] += 1
        self.assertNotEqual(reopened["revision"], plan["revision"])
        self.assertNotEqual(self.engine.plan_digest(reopened), self.engine.plan_digest(plan))

    def test_plan_digest_does_not_depend_on_dictionary_insertion_order(self):
        plan = self.engine.default_plan()
        reversed_plan = dict(reversed(list(plan.items())))
        self.assertEqual(self.engine.plan_digest(plan), self.engine.plan_digest(reversed_plan))

    def test_plan_validation_owns_nested_values_and_does_not_mutate_its_argument(self):
        plan = self.engine.default_plan()
        before = copy.deepcopy(plan)
        validated = self.engine.validate_plan(plan)
        self.assertEqual(validated, before)
        self.assertEqual(plan, before)
        self.assertIsNot(validated, plan)
        for key, value in plan.items():
            if isinstance(value, (list, dict)):
                self.assertIsNot(validated[key], value)

    def test_operation_order_and_statistical_exclusion_are_explicit(self):
        plan = self.engine.default_plan()
        self.assertEqual(plan["statistical_policy"], "none")
        self.assertEqual([step["kind"] for step in plan["steps"]], [
            "schema", "normalize", "validate_rows", "deduplicate", "reconcile",
        ])
        self.assertTrue(all(type(step["version"]) is int and step["version"] == 1 for step in plan["steps"]))
        alterations = [
            lambda item: item.update(statistical_policy="refit"),
            lambda item: item["steps"].reverse(),
            lambda item: item["steps"][0].update(version=2),
            lambda item: item["steps"].append({"kind": "missing", "version": 1, "strategy": "median"}),
            lambda item: item["steps"][1].update(sku_case="lower"),
            lambda item: item["steps"][2].update(quantity="median"),
            lambda item: item["steps"][3].update(conflict="keep_first"),
        ]
        for index, change in enumerate(alterations):
            invalid = copy.deepcopy(plan)
            change(invalid)
            with self.subTest(alteration=index), self.assertRaises(ValueError):
                self.engine.validate_plan(invalid)

    def test_plan_revision_preserves_identity_and_never_edits_the_original(self):
        plan = self.engine.default_plan()
        before = copy.deepcopy(plan)
        mapping = copy.deepcopy(plan["steps"][1]["status_mapping"])
        mapping["backordered"] = "pending"
        revised = self.engine.revise_plan(plan, status_mapping=mapping)
        self.assertEqual(plan, before)
        self.assertEqual(revised["plan_id"], plan["plan_id"])
        self.assertEqual(revised["revision"], plan["revision"] + 1)
        self.assertNotEqual(self.engine.plan_digest(revised), self.engine.plan_digest(plan))
        mapping["backordered"] = "fulfilled"
        self.assertEqual(revised["steps"][1]["status_mapping"]["backordered"], "pending")
        self.assertEqual(self.engine.loads_plan(self.engine.dumps_plan(revised)), revised)

    def test_plan_cannot_map_to_an_undeclared_canonical_status(self):
        plan = self.engine.default_plan()
        mapping = copy.deepcopy(plan["steps"][1]["status_mapping"])
        mapping["backordered"] = "arbitrary-new-output-status"
        with self.assertRaises(ValueError):
            self.engine.revise_plan(plan, status_mapping=mapping)

    def test_unknown_plan_formats_versions_and_non_integer_revisions_fail_explicitly(self):
        for key, value in (("format", "foreign-plan"), ("version", 2), ("version", True),
                           ("revision", True), ("revision", 0), ("revision", "1"),
                           ("plan_id", "not-an-identifier")):
            plan = self.engine.default_plan()
            plan[key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                self.engine.validate_plan(plan)

    def test_unknown_top_level_execution_fields_are_not_silently_ignored(self):
        plan = self.engine.default_plan()
        plan["arbitrary_python"] = "print('never execute')"
        with self.assertRaises(ValueError):
            self.engine.validate_plan(plan)

    def test_non_json_objects_and_duplicate_json_keys_are_rejected(self):
        plan = self.engine.default_plan()
        plan["arbitrary_python"] = lambda: None
        with self.assertRaises(ValueError):
            self.engine.dumps_plan(plan)
        with self.assertRaises(ValueError):
            self.engine.loads_plan(b'{"format":"x","format":"y"}')

    def test_valid_plan_json_rejects_duplicate_keys_nonfinite_values_and_bool_versions(self):
        plan = self.engine.default_plan()
        document = json.dumps(plan)
        duplicates = [
            ('{"revision": 1,' + document[1:]).encode("utf-8"),
            document.replace('"kind": "schema"', '"kind": "schema", "kind": "schema"', 1).encode("utf-8"),
        ]
        for data in duplicates:
            with self.subTest(kind="duplicate", data=data), self.assertRaises(ValueError):
                self.engine.loads_plan(data)
        for value in (float("nan"), float("inf"), -float("inf"), True, 1.0):
            invalid = copy.deepcopy(plan)
            invalid["revision"] = value
            with self.subTest(revision=value), self.assertRaises(ValueError):
                self.engine.loads_plan(json.dumps(invalid).encode("utf-8"))
        invalid = copy.deepcopy(plan)
        invalid["steps"][0]["version"] = True
        with self.assertRaises(ValueError):
            self.engine.loads_plan(json.dumps(invalid).encode("utf-8"))

    def test_plan_loader_rejects_over_budget_json_before_accepting_a_valid_payload(self):
        document = json.dumps(self.engine.default_plan()).encode("utf-8")
        with self.assertRaises(ValueError):
            self.engine.loads_plan(b" " * (1024 * 1024) + document)

    def test_ingestion_records_raw_digest_and_stable_source_scoped_row_identifiers(self):
        import hashlib
        data = (FIXTURES / "month_a.csv").read_bytes()
        source = self.engine.ingest_csv(data)
        self.assertEqual(source["raw_sha256"], hashlib.sha256(data).hexdigest())
        self.assertRegex(source["source_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(source["row_ids"], [f"{source['source_id']}:{index}" for index in range(1, 13)])
        self.assertTrue(all(type(value) is str for row in source["rows"] for value in row))
        self.assertEqual(source["rows"][0][:2], ["000101", "01"])
        self.assertEqual(source["rows"][10][4], "")

    def test_source_validation_rejects_tampered_rows_hashes_and_row_identities(self):
        source = self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes())
        validated = self.engine.validate_source(source)
        self.assertEqual(validated, source)
        self.assertIsNot(validated, source)
        self.assertIsNot(validated["rows"], source["rows"])
        mutations = [
            lambda item: item["rows"][0].__setitem__(0, "tampered-order"),
            lambda item: item.update(raw_sha256="0" * 64),
            lambda item: item["row_ids"].__setitem__(0, "forged-source:1"),
            lambda item: item.update(source_id="0" * 64),
        ]
        for index, mutate in enumerate(mutations):
            tampered = copy.deepcopy(source)
            mutate(tampered)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self.engine.validate_source(tampered)

    def test_frozen_fixture_manifest_still_matches_every_original_fixture(self):
        import hashlib
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            with self.subTest(path=item["path"]):
                data = (FIXTURES / item["path"]).read_bytes()
                self.assertEqual(len(data), item["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])

    def test_duplicate_header_and_malformed_csv_fail_before_execution(self):
        invalid = [
            b"order_id,order_id\n000001,01\n",
            b'order_id,line_id\n"unterminated,01\n',
            b"order_id,line_id\n000001,01,unexpected\n",
            b"order_id,line_id\n000001\n",
            b"",
            b"order_id,line_id\n\xff,01\n",
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.engine.ingest_csv(data)

    def test_only_explicit_single_character_delimiters_are_accepted(self):
        data = (FIXTURES / "month_a.csv").read_bytes()
        for delimiter in ("", "||", "\n", '"', None):
            with self.subTest(delimiter=delimiter), self.assertRaises(ValueError):
                self.engine.ingest_csv(data, delimiter=delimiter)

    def test_delimited_source_options_do_not_depend_on_global_widget_state(self):
        data = (FIXTURES / "month_b.csv").read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO(data)))
        stream = io.StringIO(newline="")
        csv.writer(stream, delimiter=";", lineterminator="\n").writerows(rows)
        source = self.engine.ingest_csv(stream.getvalue().encode("utf-8"), delimiter=";", encoding="utf-8-sig")
        result = self.engine.execute(source, self.engine.default_plan())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"]["accepted_rows"], 5)

    def test_source_identity_includes_read_options_even_when_raw_bytes_match(self):
        data = (FIXTURES / "month_a.csv").read_bytes()
        plain = self.engine.ingest_csv(data, encoding="utf-8")
        bom_aware = self.engine.ingest_csv(data, encoding="utf-8-sig")
        self.assertEqual(plain["raw_sha256"], bom_aware["raw_sha256"])
        self.assertEqual(plain["rows"], bom_aware["rows"])
        self.assertNotEqual(plain["source_id"], bom_aware["source_id"])
        self.assertFalse(set(plain["row_ids"]) & set(bom_aware["row_ids"]))

    def test_empty_physical_records_are_skipped_but_blank_cell_records_are_preserved(self):
        data = b"order_id,line_id,order_date,sku,quantity,fulfillment_status\n\n,,,,,\n\n"
        source = self.engine.ingest_csv(data)
        self.assertEqual(source["rows"], [[""] * 6])
        self.assertEqual(source["row_ids"], [source["source_id"] + ":1"])
        result = self.engine.execute(source, self.engine.default_plan())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"]["source_rows"], 1)
        self.assertEqual(result["report"]["rejected_rows"], 1)
        self.assertEqual(set(result["rejected"][0]["reasons"]), {
            "missing_order_id", "missing_line_id", "missing_order_date",
            "missing_sku", "missing_quantity", "missing_fulfillment_status",
        })

    def test_blank_headers_and_unknown_encoding_fail_without_creating_a_source(self):
        for data in (b"order_id,\n000001,01\n", b"order_id,   \n000001,01\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.engine.ingest_csv(data)
        with self.assertRaises(ValueError):
            self.engine.ingest_csv((FIXTURES / "month_a.csv").read_bytes(), encoding="not-a-real-codec")

    def test_ingestion_enforces_central_upload_row_column_cell_and_storage_budgets(self):
        from backend import resources
        data = (FIXTURES / "month_a.csv").read_bytes()
        limits = {
            "MAX_UPLOAD_BYTES": len(data) - 1,
            "MAX_DATASET_ROWS": 1,
            "MAX_DATASET_COLUMNS": 5,
            "MAX_DATASET_CELLS": 6,
            "MAX_DATASET_BYTES": 1,
        }
        for field, limit in limits.items():
            with self.subTest(budget=field), patch.object(resources, field, limit), self.assertRaises(ValueError):
                self.engine.ingest_csv(data)
        header_only = data.splitlines(keepends=True)[0]
        with self.subTest(budget="header-only source storage"), patch.object(resources, "MAX_DATASET_BYTES", 1), self.assertRaises(ValueError):
            self.engine.ingest_csv(header_only)


if __name__ == "__main__":
    unittest.main()
