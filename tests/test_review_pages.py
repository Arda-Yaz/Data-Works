import unittest

from streamlit.testing.v1 import AppTest


UPLOAD_SCRIPT = '''
import pandas as pd
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_upload import data_upload

StateManager.init_session()
if not StateManager.has_dataset():
    df = pd.DataFrame({"value": ["1", "2", "3"], "code": ["A001", "A002", "A003"]})
    StateManager.load_dataset(df, infer_column_meta(df))
data_upload()
'''


class PageTests(unittest.TestCase):
    def test_schema_preview_and_confirmation_commit_the_same_values(self):
        app = AppTest.from_string(UPLOAD_SCRIPT).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["df"]["value"].tolist(), ["1", "2", "3"])
        confirm = next(button for button in app.button if button.label == "Confirm Schema")
        confirm.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["schema_confirmed"])
        self.assertEqual(app.session_state["df"]["value"].tolist(), [1, 2, 3])
        self.assertEqual(app.session_state["df"]["code"].tolist(), ["A001", "A002", "A003"])

    def test_invalid_schema_is_reported_and_cannot_be_committed(self):
        script = UPLOAD_SCRIPT.replace(
            'StateManager.load_dataset(df, infer_column_meta(df))',
            'meta = infer_column_meta(df)\n'
            '    meta["code"]["user_dtype"] = "int64"\n'
            '    StateManager.load_dataset(df, meta)',
        )
        app = AppTest.from_string(script).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("code", app.error[0].value)
        confirm = next(button for button in app.button if button.label == "Confirm Schema")
        self.assertTrue(confirm.disabled)
        self.assertFalse(app.session_state["schema_confirmed"])
        self.assertEqual(app.session_state["df"]["value"].tolist(), ["1", "2", "3"])

    def test_training_page_keeps_rows_with_missing_features(self):
        app = AppTest.from_string('''
import numpy as np
import pandas as pd
from backend.ingestion import infer_column_meta
from backend.state import StateManager
from frontend.pages.data_ml import data_ml

StateManager.init_session()
if not StateManager.has_dataset():
    df = pd.DataFrame({"target": np.arange(40, dtype=float), "value": [1.0, None, 2.0, None] * 10})
    StateManager.load_dataset(df, infer_column_meta(df))
data_ml()
''').run()
        self.assertEqual(len(app.exception), 0)
        app.multiselect(key="ml_models").set_value(["Linear Regression"])
        app.button(key="ml_train").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("Using 40 of 40 rows" in caption.value for caption in app.caption))
        self.assertTrue(any(table.value.index.tolist() == ["Linear Regression"] for table in app.dataframe))


if __name__ == "__main__":
    unittest.main()
