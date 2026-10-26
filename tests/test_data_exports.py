import io
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from backend.exports import export_csv


class CSVExportTests(unittest.TestCase):
    def test_raw_csv_preserves_text_values_and_quotes_correctly(self):
        frame = pd.DataFrame({"text": ["0001", "NA", '=SUM(A1:A2)', 'comma, quote"\nline'], "value": [1, 2, 3, 4]})
        result = pd.read_csv(io.BytesIO(export_csv(frame, spreadsheet_safe=False)), dtype=object, keep_default_na=False)
        self.assertEqual(result["text"].tolist(), frame["text"].tolist())

    def test_safe_csv_escapes_formula_like_text_but_not_numeric_values(self):
        values = ["=1+1", "+SUM(A1)", "-2+3", "@cmd", "  =1", "\ttext", "normal", "0001"]
        frame = pd.DataFrame({"text": values, "number": [-1] * len(values)})
        before = frame.copy(deep=True)
        result = pd.read_csv(io.BytesIO(export_csv(frame)), dtype=object, keep_default_na=False)
        self.assertEqual(result["text"].tolist(), ["'" + value for value in values[:6]] + values[6:])
        self.assertEqual(result["number"].tolist(), ["-1"] * len(values))
        assert_frame_equal(frame, before)

    def test_safe_csv_handles_category_headers_and_index_text(self):
        frame = pd.DataFrame({"=header": pd.Categorical(["=formula", "safe"]), "flag": [True, False]}, index=pd.Index(["@id", "normal"], name="=index"))
        text = export_csv(frame, include_index=True).decode("utf-8-sig")
        self.assertIn("'=header", text)
        self.assertIn("'=index", text)
        self.assertIn("'@id", text)
        self.assertIn("'=formula", text)

    def test_unicode_and_empty_exports_are_valid_utf8_csv(self):
        frame = pd.DataFrame({"city": ["İstanbul", "東京"]})
        result = pd.read_csv(io.BytesIO(export_csv(frame)))
        assert_frame_equal(result, frame)
        self.assertIsInstance(export_csv(pd.DataFrame()), bytes)

    def test_safe_export_escapes_bom_prefixed_formulas_and_multi_index_headers(self):
        frame = pd.DataFrame([["\ufeff =1+1"]], columns=pd.MultiIndex.from_tuples([("=outer", "@inner")], names=["level", "field"]))
        text = export_csv(frame).decode("utf-8-sig")
        self.assertIn("'=outer", text)
        self.assertIn("'@inner", text)
        self.assertIn("'\ufeff =1+1", text)

    def test_invalid_arguments_are_actionable(self):
        for data, kwargs in ((None, {}), (pd.DataFrame(), {"spreadsheet_safe": "yes"}), (pd.DataFrame(), {"include_index": 1})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                export_csv(data, **kwargs)


if __name__ == "__main__":
    unittest.main()
