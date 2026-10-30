import copy
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from backend.dataset_preparation import prepare_dataset


class PreparationBoundaryTests(unittest.TestCase):
    def test_preparation_owns_frame_and_nested_metadata_and_preserves_index(self):
        frame = pd.DataFrame({"code": ["001", "002", "001"]}, index=[4, 4, 7], dtype=object)
        metadata = {"code": {"role": "id", "allowed_values": ["001", "002"]}}
        original = frame.copy(deep=True)
        original_metadata = copy.deepcopy(metadata)

        candidate, synchronized = prepare_dataset(frame, metadata)

        assert_frame_equal(frame, original)
        self.assertEqual(metadata, original_metadata)
        self.assertEqual(candidate.index.tolist(), [4, 4, 7])
        self.assertEqual(synchronized["code"]["role"], "id")
        self.assertEqual(synchronized["code"]["cardinality"], 2)
        self.assertEqual(synchronized["code"]["current_dtype"], "object")
        candidate.iloc[0, 0] = "999"
        synchronized["code"]["allowed_values"].append("999")
        assert_frame_equal(frame, original)
        self.assertEqual(metadata, original_metadata)

    def test_string_storage_and_missing_semantics_survive_preparation(self):
        for missing in (pd.NA, float("nan")):
            with self.subTest(missing=repr(missing)):
                storage = pd.StringDtype(storage="python", na_value=missing)
                frame = pd.DataFrame({"code": pd.Series(["001", None, "002"], dtype=storage)})
                frame.index = [4, 4, 7]
                before = frame.copy(deep=True)
                candidate, metadata = prepare_dataset(frame, {})
                assert_frame_equal(candidate, before)
                self.assertEqual(metadata["code"]["current_dtype"], str(storage))
                self.assertEqual(metadata["code"]["original_dtype"], str(storage))
                self.assertEqual(metadata["code"]["inferred_dtype"], "object")
                self.assertEqual(candidate.index.tolist(), [4, 4, 7])
                candidate.iloc[0, 0] = "999"
                assert_frame_equal(frame, before)
    def test_dtype_change_clears_conversion_request_unless_schema_is_explicit(self):
        before = pd.DataFrame({"value": pd.Series([1, 2], dtype="Int64")})
        after = before.astype({"value": "Float64"})
        metadata = {"value": {"inferred_dtype": "int64", "user_dtype": "int64"}}
        _, implicit = prepare_dataset(after, metadata, previous_df=before)
        _, explicit = prepare_dataset(after, metadata, previous_df=before, explicit_schema=True)
        self.assertIsNone(implicit["value"]["user_dtype"])
        self.assertEqual(implicit["value"]["inferred_dtype"], "float64")
        self.assertEqual(explicit["value"]["user_dtype"], "int64")

    def test_invalid_candidate_does_not_mutate_frame_or_schema(self):
        frame = pd.DataFrame({"value": [1, 2]}, index=[3, 3])
        metadata = {"value": {"role": "unsupported", "allowed_values": ["1"]}}
        original = frame.copy(deep=True)
        original_metadata = copy.deepcopy(metadata)
        with self.assertRaisesRegex(ValueError, "role"):
            prepare_dataset(frame, metadata)
        assert_frame_equal(frame, original)
        self.assertEqual(metadata, original_metadata)

    def test_replay_runs_in_fresh_process_that_rejects_streamlit_imports(self):
        source = textwrap.dedent("""
            import importlib.abc
            import sys

            class RejectStreamlit(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'streamlit' or fullname.startswith('streamlit.'):
                        raise AssertionError('Replay imported Streamlit: ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectStreamlit())
            import pandas as pd
            from backend.ingestion import infer_column_meta
            from backend.operations import replay_operations

            original = pd.DataFrame({'x': ['1', None, '3']}, index=[4, 4, 7], dtype=object)
            metadata = infer_column_meta(original)
            schema = {key: dict(value) for key, value in metadata.items()}
            schema['x']['user_dtype'] = 'int64'
            operations = [
                {'kind': 'schema', 'parameters': {'column_meta': schema}},
                {'kind': 'missing', 'parameters': {'strategies': {'x': 'median'}}},
            ]
            result = replay_operations(original, metadata, operations)
            assert result['df']['x'].tolist() == [1, 2, 3]
            assert result['df'].index.tolist() == [4, 4, 7]
            assert result['schema_confirmed'] is True
            assert result['statistical_operations'] == ['Handle missing values']
            assert original['x'].tolist() == ['1', None, '3']
            assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
            print('UI-independent replay passed')
        """)
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("UI-independent replay passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
