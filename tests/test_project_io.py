import io
import json
import unittest
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from backend.project_io import (
    FORMAT, FORMAT_VERSION, decode_value, dumps_project, encode_value, loads_project,
)


class ProjectCodecTests(unittest.TestCase):
    @staticmethod
    def archive(envelope, filename="project.json", extra=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as output:
            output.writestr(filename, json.dumps(envelope))
            if extra:
                output.writestr(extra, "unused")
        return buffer.getvalue()

    def test_nullable_and_time_dtypes_round_trip_without_precision_loss(self):
        frame = pd.DataFrame({
            "whole": pd.Series([9007199254740993, None, -2], dtype="Int64"),
            "unsigned": pd.Series([2**64 - 1, None, 0], dtype="UInt64"),
            "decimal": pd.Series([1.25, None, -0.0], dtype="Float64"),
            "narrow": pd.Series([1.25, None, 2.5], dtype="Float32"),
            "flag": pd.Series([True, None, False], dtype="boolean"),
            "text": pd.Series(["0001", None, "NA"], dtype="string"),
            "when": pd.to_datetime(["2024-01-01", None, "2024-02-01"]).tz_localize("Europe/Istanbul"),
            "duration": pd.to_timedelta([1, None, 2], unit="D"),
            "category": pd.Categorical.from_codes([0, -1, 1], categories=["b", "a", "unused"], ordered=True),
        })
        frame.index = pd.Index(["same", "same", "last"], name="record")
        result = loads_project(dumps_project({"df": frame}))["df"]
        assert_frame_equal(result, frame)
        self.assertEqual(int(result.iloc[0, 0]), 9007199254740993)

    def test_string_missing_semantics_and_storage_survive_frame_and_index_round_trip(self):
        for storage in ("python", "pyarrow"):
            for missing in (pd.NA, np.nan):
                dtype = pd.StringDtype(storage=storage, na_value=missing)
                with self.subTest(storage=storage, missing=str(missing)):
                    frame = pd.DataFrame({"token": pd.Series(["0001", None, "NA"], dtype=dtype)})
                    frame.index = pd.Index(["first", None, "last"], dtype=dtype, name="source")
                    frame.columns = pd.Index(["token"], dtype=dtype)
                    restored = loads_project(dumps_project({"df": frame}))["df"]
                    assert_frame_equal(restored, frame)
                    self.assertEqual(restored["token"].dtype.storage, storage)
                    self.assertEqual(restored["token"].dtype.na_value is pd.NA, missing is pd.NA)
                    if missing is pd.NA:
                        self.assertIs(restored.iloc[1, 0], pd.NA)
                    else:
                        self.assertTrue(np.isnan(restored.iloc[1, 0]))

    def test_object_string_indexes_do_not_reinfer_a_new_string_dtype(self):
        frame = pd.DataFrame({"code": pd.Series(["001", None], dtype=object)})
        frame.index = pd.Index(["one", None], dtype=object)
        frame.columns = pd.Index(["code"], dtype=object)
        assert_frame_equal(decode_value(encode_value(frame)), frame)

    def test_legacy_string_descriptor_retains_nullable_semantics_and_invalid_mode_rejects(self):
        series = pd.Series(["001", None], dtype=pd.StringDtype(storage="python"))
        encoded = encode_value(series)
        self.assertEqual(encoded["data"]["dtype"], {"kind": "string", "storage": "python"})
        self.assertIs(decode_value(encoded).iloc[1], pd.NA)
        encoded["data"]["dtype"]["na_value"] = "guess"
        with self.assertRaises(ValueError):
            decode_value(encoded)

    def test_large_integer_categorical_levels_are_not_converted_to_floats(self):
        value = 2**53 + 1
        frame = pd.DataFrame({"category": pd.Categorical.from_codes([0, -1], categories=pd.Index([value], dtype="int64"))})
        result = decode_value(encode_value(frame))
        assert_frame_equal(result, frame)
        self.assertEqual(int(result.loc[0, "category"]), value)

    def test_mixed_column_labels_and_mapping_keys_are_preserved(self):
        frame = pd.DataFrame([[1, "1", 2.5]], columns=pd.Index([1, "1", ("nested", 2)], dtype=object))
        payload = {"df": frame, "column_meta": {1: {"role": "feature"}, "1": {"role": "id"}, ("nested", 2): {"role": "ignore"}}}
        result = loads_project(dumps_project(payload))
        assert_frame_equal(result["df"], frame)
        self.assertEqual(result["column_meta"], payload["column_meta"])

    def test_object_scalar_types_and_missing_markers_are_preserved(self):
        values = [None, pd.NA, pd.NaT, np.nan, "NaN", 2**64, Decimal("1.2300"), Fraction(1, 3),
                  date(2024, 1, 2), datetime(2024, 2, 3, tzinfo=ZoneInfo("Europe/Istanbul")),
                  timedelta(days=2), b"bytes", 1 + 2j, {"nested": [1, "two"]}]
        frame = pd.DataFrame({"objects": pd.Series(values, dtype=object)})
        result = loads_project(dumps_project({"df": frame}))["df"]
        assert_frame_equal(result, frame)
        self.assertIs(result.iloc[1, 0], pd.NA)
        self.assertIs(result.iloc[2, 0], pd.NaT)
        self.assertIsInstance(result.iloc[4, 0], str)

    def test_series_arrays_and_reserved_mapping_tags_round_trip(self):
        series = pd.Series([1, None], dtype="Int64", index=["a", "a"], name="truth")
        arrays = [np.array([1.0, np.nan]), np.array([2**63, 2**64 - 1], dtype="uint64"),
                  np.array(["2024-01-01", "NaT"], dtype="datetime64[D]"), np.array(3, dtype="int32")]
        payload = {"series": series, "arrays": arrays, "escaped": {"$dataworks": "not a tag"}}
        result = loads_project(dumps_project(payload))
        assert_series_equal(result["series"], series)
        for old, new in zip(arrays, result["arrays"]):
            np.testing.assert_array_equal(old, new)
            self.assertEqual(old.dtype, new.dtype)
        self.assertEqual(result["escaped"], payload["escaped"])

    def test_empty_tables_and_named_range_index_round_trip(self):
        frames = [pd.DataFrame(index=pd.RangeIndex(10, 16, 2, name="rows")),
                  pd.DataFrame({"empty": pd.Series([], dtype="Int64")}), pd.DataFrame()]
        for frame in frames:
            with self.subTest(frame=frame.shape):
                assert_frame_equal(loads_project(dumps_project({"df": frame}))["df"], frame)

    def test_multi_index_levels_and_datetime_frequency_are_preserved(self):
        index = pd.MultiIndex(levels=[pd.Index(["a", "b", "unused"]), pd.Index([1, 2])],
                              codes=[[0, 0, 1], [0, 0, 1]], names=["group", "item"])
        frame = pd.DataFrame({"value": [1, 2, 3]}, index=index)
        assert_frame_equal(decode_value(encode_value(frame)), frame)
        timed = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="when"))
        assert_frame_equal(decode_value(encode_value(timed)), timed)

    def test_period_and_interval_values_use_data_only_constructors(self):
        frame = pd.DataFrame({"period": pd.period_range("2024-01", periods=3, freq="M"),
                              "interval": pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])})
        assert_frame_equal(decode_value(encode_value(frame)), frame)

    def test_custom_objects_and_executable_serialization_are_rejected(self):
        class Unsafe:
            def __reduce__(self):
                raise AssertionError("Executable serialization must never be attempted")
        with self.assertRaisesRegex(ValueError, "custom objects"):
            dumps_project({"model": Unsafe()})
        with self.assertRaises(ValueError):
            loads_project(b"\x80\x04pickle data")
        with self.assertRaises(ValueError):
            decode_value({"$dataworks": "python", "module": "os", "call": "system"})

    def test_invalid_envelopes_and_zip_members_are_rejected(self):
        valid = {"format": FORMAT, "version": FORMAT_VERSION, "payload": {}}
        cases = [self.archive({**valid, "version": 999}), self.archive({**valid, "version": True}),
                 self.archive(valid, filename="../project.json"), self.archive(valid, extra="model.pkl"),
                 self.archive({**valid, "format": "foreign"})]
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                loads_project(data)

    def test_limits_are_enforced_on_save_and_load(self):
        data = dumps_project({"text": "some data"})
        with patch("backend.project_io.MAX_ARCHIVE_BYTES", len(data) - 1), self.assertRaises(ValueError):
            loads_project(data)
        with patch("backend.project_io.MAX_JSON_BYTES", 4), self.assertRaises(ValueError):
            loads_project(data)
        with patch("backend.project_io.MAX_DATA_ITEMS", 2), self.assertRaises(ValueError):
            dumps_project({"df": pd.DataFrame({"x": [1, 2, 3]})})

    def test_malformed_shapes_dtypes_and_missing_lengths_are_rejected(self):
        encoded = encode_value(pd.DataFrame({"x": [1, 2]}))
        encoded["data"][0]["values"].pop()
        with self.assertRaises(ValueError):
            decode_value(encoded)
        for dtype in ("V100", "U999999999", "object;exec", "f16"):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                decode_value({"$dataworks": "array", "dtype": dtype, "shape": [1], "values": [1]})

    def test_depth_and_duplicate_json_keys_are_rejected(self):
        value = None
        for _ in range(105):
            value = [value]
        with self.assertRaises(ValueError):
            encode_value(value)
        raw = b'{"format":"dataworks-project","version":1,"version":1,"payload":{}}'
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("project.json", raw)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            loads_project(output.getvalue())

    def test_declared_allocations_have_a_total_budget(self):
        payload = [{"$dataworks": "numpy_scalar", "dtype": "U10", "value": "x"}] * 2
        with patch("backend.project_io.MAX_JSON_BYTES", 60), self.assertRaisesRegex(ValueError, "allocation limit"):
            decode_value(payload)

    def test_declared_narrow_float_cannot_silently_round_a_value(self):
        encoded = encode_value(pd.DataFrame({"x": pd.Series([1.0], dtype="Float32")}))
        encoded["data"][0]["values"] = [16777217]
        with self.assertRaisesRegex(ValueError, "round"):
            decode_value(encoded)

    def test_numpy_arrays_reject_lossy_or_wrong_type_payloads(self):
        cases = [("int64", 1.5), ("uint8", -1), ("bool", "false"),
                 ("float32", 16777217), ("complex64", complex(16777217, 1)),
                 ("U1", "ABC"), ("S1", b"ABC"),
                 ("datetime64[D]", np.datetime64("2026-10-30T12", "h"))]
        for dtype, value in cases:
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                decode_value({"$dataworks": "array", "dtype": np.dtype(dtype).str,
                              "shape": [1], "values": [encode_value(value)]})

    def test_numpy_scalar_tag_rejects_array_values_and_lossy_casts(self):
        for dtype, value in (("int64", [1, 2]), ("float64", [[1.0]]),
                             ("object", [1, 2]), ("int64", 1.5), ("float32", 16777217)):
            with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                decode_value({"$dataworks": "numpy_scalar", "dtype": np.dtype(dtype).str,
                              "value": encode_value(value)})

    def test_valid_numpy_arrays_keep_dtype_values_and_nonfinite_markers(self):
        arrays = [np.array([True, False]), np.array([-128, 127], dtype="int8"),
                  np.array([1.5, np.nan, np.inf, -np.inf], dtype="float16"),
                  np.array([1+2j, complex(np.nan, 1), complex(1, np.nan)], dtype="complex64"),
                  np.array(["001", "NA"], dtype="U3"), np.array([b"abc", b""], dtype="S3"),
                  np.array([1, "NaT"], dtype="timedelta64[ns]"),
                  np.array(np.uint64(2**64 - 1))]
        objects = np.empty(2, dtype=object)
        objects[0], objects[1] = {"kept": [1, 2]}, ["nested", None]
        arrays.append(objects)
        for source in arrays:
            with self.subTest(dtype=source.dtype):
                result = decode_value(encode_value(source))
                self.assertEqual(result.dtype, source.dtype)
                self.assertEqual(result.shape, source.shape)
                np.testing.assert_array_equal(result, source)


if __name__ == "__main__":
    unittest.main()
