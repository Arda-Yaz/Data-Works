from io import BytesIO, StringIO
from struct import pack
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from backend.ingestion import (
    apply_inferred_dtypes, infer_column_meta, infer_dtype, list_excel_sheets,
    parse_file, validate_numeric_format,
)


def upload(contents, name="data.csv"):
    stream = BytesIO(contents) if isinstance(contents, bytes) else StringIO(contents)
    stream.name = name
    return stream


class FileOptionsTests(unittest.TestCase):
    def test_delimiter_and_encoding_preserve_literal_text_and_cursor(self):
        stream = upload("code;city;missing\n0001;İzmir;NA\n0002;Ankara;\n".encode("cp1254"))
        stream.seek(5)
        frame = parse_file(stream, delimiter=";", encoding="cp1254")
        self.assertEqual(stream.tell(), 5)
        self.assertEqual(frame["code"].tolist(), ["0001", "0002"])
        self.assertEqual(frame["city"].tolist(), ["İzmir", "Ankara"])
        self.assertEqual(frame.loc[0, "missing"], "NA")
        self.assertTrue(pd.isna(frame.loc[1, "missing"]))

    def test_tsv_default_and_literal_pipe_separator(self):
        self.assertEqual(parse_file(upload("id\tlabel\n001\tNULL\n", "data.tsv")).loc[0, "id"], "001")
        frame = parse_file(upload('id|label\n001|"a|b"\n', "data.txt"), delimiter="|")
        self.assertEqual(frame.loc[0, "label"], "a|b")

    def test_invalid_separator_or_encoding_fails_explicitly(self):
        for delimiter in ("", "||", "\n", '"', "\0"):
            with self.subTest(delimiter=delimiter), self.assertRaisesRegex(ValueError, "delimiter"):
                parse_file(upload("a,b\n1,2\n"), delimiter=delimiter)
        with self.assertRaisesRegex(ValueError, "encoding"):
            parse_file(upload(b"name\n\xff\n"), encoding="nonexistent-encoding")
        with self.assertRaisesRegex(ValueError, "decoded"):
            parse_file(upload(b"name\n\xff\n"))

    def test_duplicate_headers_extra_cells_and_invalid_quotes_are_rejected(self):
        for text, message in (
            ("a,a\n1,2\n", "duplicate"),
            ("a,b\n1,2,3\n", "more fields"),
            ('a,b\n1,"unterminated\n', "quoting"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                parse_file(upload(text))

    def workbook(self):
        stream = BytesIO()
        with pd.ExcelWriter(stream, engine="openpyxl") as writer:
            pd.DataFrame({"id": ["0001"], "value": [2.5]}).to_excel(writer, sheet_name="First", index=False)
            pd.DataFrame({"id": ["0002"], "value": [7.5]}).to_excel(writer, sheet_name="Second", index=False)
        stream.name = "example.xlsx"
        stream.seek(11)
        return stream

    def test_excel_sheet_listing_and_selection_preserve_upload_cursor(self):
        stream = self.workbook()
        self.assertEqual(list_excel_sheets(stream), ["First", "Second"])
        self.assertEqual(stream.tell(), 11)
        chosen = parse_file(stream, sheet_name="Second")
        self.assertEqual(chosen.loc[0, "id"], "0002")
        self.assertEqual(chosen.loc[0, "value"], 7.5)
        self.assertEqual(stream.tell(), 11)
        self.assertEqual(parse_file(stream).loc[0, "id"], "0001")

    def test_excel_duplicate_headers_are_rejected_before_pandas_mangling(self):
        stream = BytesIO()
        pd.DataFrame([["x", "x"], [1, 2]]).to_excel(stream, header=False, index=False)
        stream.name = "duplicate.xlsx"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_file(stream)

    def test_xls_legacy_workbook_reads_with_installed_engine(self):
        # Minimal BIFF2 worksheet built in memory: headers, a text ID, and a number.
        def record(opcode, payload=b""):
            return pack("<HH", opcode, len(payload)) + payload

        parts = [record(0x0009, pack("<HH", 2, 0x0010)), record(0x0042, pack("<H", 1252))]
        for row, column, text in ((0, 0, "id"), (0, 1, "value"), (1, 0, "0001")):
            encoded = text.encode("cp1252")
            parts.append(record(0x0004, pack("<HH3sB", row, column, b"\0\0\0", len(encoded)) + encoded))
        parts.append(record(0x0003, pack("<HH3sd", 1, 1, b"\0\0\0", 2.5)))
        parts.append(record(0x000A))
        stream = upload(b"".join(parts), "legacy.xls")
        self.assertEqual(list_excel_sheets(stream), ["Sheet 1"])
        result = parse_file(stream, sheet_name="Sheet 1")
        self.assertEqual(result.loc[0, "id"], "0001")
        self.assertEqual(result.loc[0, "value"], 2.5)

    def test_nested_json_path_preserves_large_integer_and_quoted_identifier(self):
        text = '{"response":{"records":[{"id":"0001","n":9007199254740993},{"id":"0002","n":null}]}}'
        frame = parse_file(upload(text, "data.json"), json_path=["response", "records"])
        self.assertEqual(frame.loc[0, "id"], "0001")
        self.assertEqual(frame.loc[0, "n"], 9007199254740993)
        self.assertIsInstance(frame.loc[0, "n"], int)
        self.assertTrue(pd.isna(frame.loc[1, "n"]))

    def test_json_lines_and_column_oriented_json(self):
        frame = parse_file(upload('{"id":"001","flag":true}\n\n{"id":"002","flag":false}\n', "data.json"), json_lines=True)
        self.assertEqual(frame["id"].tolist(), ["001", "002"])
        self.assertEqual(frame["flag"].tolist(), [True, False])
        columns = parse_file(upload('{"id":["001","002"],"n":[9007199254740993,null]}', "columns.json"))
        self.assertEqual(columns.loc[0, "n"], 9007199254740993)
        self.assertEqual(parse_file(upload('{"id":"001"}\n', "data.jsonl")).loc[0, "id"], "001")

    def test_json_invalid_path_structure_and_duplicates_are_explicit(self):
        cases = [
            ('{"records":[]}', {"json_path": ["missing"]}),
            ('[{"a":{"nested":1}}]', {}),
            ('[1,2,3]', {}),
            ('{"a":[1],"b":[1,2]}', {}),
            ('{"a":1,"a":2}', {}),
            ('[{"a":NaN}]', {}),
            ('[{"a":1e999}]', {}),
            ('[{"a":1e-999}]', {}),
            ('{"a":1}', {"json_path": "a"}),
            ('{"a":1}', {"json_path": ["a"], "json_lines": True}),
        ]
        for text, options in cases:
            with self.subTest(text=text, options=options), self.assertRaises(ValueError):
                parse_file(upload(text, "data.json"), **options)


class NumericConventionTests(unittest.TestCase):
    def test_localized_numbers_are_only_converted_during_schema_application(self):
        raw = pd.DataFrame({"amount": ["1.234,50", "-2.345,75", None]})
        before = raw.copy(deep=True)
        metadata = infer_column_meta(raw, decimal=",", thousands=".")
        self.assertEqual(metadata["amount"]["inferred_dtype"], "float64")
        self.assertEqual(metadata["amount"]["numeric_decimal"], ",")
        self.assertEqual(metadata["amount"]["numeric_thousands"], ".")
        converted = apply_inferred_dtypes(raw, metadata)
        self.assertEqual(converted["amount"].iloc[:2].tolist(), [1234.5, -2345.75])
        self.assertTrue(pd.isna(converted.loc[2, "amount"]))
        assert_frame_equal(raw, before)

    def test_grouping_validation_rejects_malformed_numbers_atomically(self):
        for value in ("12.34,50", "1.23.456", "1.234.56", "1,2,3", "1.234 kg", "5 out of 10"):
            raw = pd.DataFrame({"amount": [value]})
            metadata = infer_column_meta(raw, decimal=",", thousands=".")
            self.assertEqual(metadata["amount"]["inferred_dtype"], "object")
            metadata["amount"]["user_dtype"] = "float64"
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "amount"):
                apply_inferred_dtypes(raw, metadata)
            self.assertEqual(raw.loc[0, "amount"], value)

    def test_native_numbers_and_mixed_json_cells_ignore_textual_conventions(self):
        frame = pd.DataFrame({"amount": pd.Series([1.5, "2.345,50", 1000, None], dtype=object)})
        metadata = infer_column_meta(frame, decimal=",", thousands=".")
        result = apply_inferred_dtypes(frame, metadata)
        self.assertEqual(result["amount"].iloc[:3].tolist(), [1.5, 2345.5, 1000.0])

    def test_space_groups_scientific_notation_and_identifiers(self):
        frame = pd.DataFrame({"amount": ["1 234,5", "-2,5e2"], "customer_id": ["0001", "0002"]})
        result = apply_inferred_dtypes(frame, infer_column_meta(frame, decimal=",", thousands=" "))
        self.assertEqual(result["amount"].tolist(), [1234.5, -250.0])
        self.assertEqual(result["customer_id"].tolist(), ["0001", "0002"])
        self.assertEqual(infer_dtype(pd.Series(["0001", "0002"]), decimal=",")[0], "object")

    def test_separator_configuration_is_validated_even_for_native_columns(self):
        for decimal, thousands in ((";", None), (".", "."), (",", ","), (".", "_")):
            with self.subTest(decimal=decimal, thousands=thousands), self.assertRaises(ValueError):
                validate_numeric_format(decimal, thousands)
            with self.assertRaises(ValueError):
                infer_column_meta(pd.DataFrame({"amount": [1]}), decimal=decimal, thousands=thousands)


if __name__ == "__main__":
    unittest.main()
