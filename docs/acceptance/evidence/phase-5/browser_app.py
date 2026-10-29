"""Local browser smoke harness: real pages, labeled synthetic fixture transport."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import streamlit as st

from backend.fulfillment import ingest_csv
from backend.state import StateManager
from frontend.pages.data_fulfillment import data_fulfillment
from frontend.pages.data_project import data_project


st.set_page_config(page_title="DataWorks synthetic workspace check", layout="wide")
StateManager.init_session()
st.sidebar.caption("Synthetic browser smoke harness. No operator observation or customer data.")
page = st.sidebar.selectbox("Smoke page", ["Order Fulfillment", "Project & Exports"])
fixture = st.sidebar.selectbox("Synthetic source", [
    "month_a.csv", "month_b.csv", "month_c_unseen_status.csv", "month_c_conflicting_duplicates.csv",
])


def load_fixture():
    StateManager.set_fulfillment_source(ingest_csv((ROOT / "tests/fixtures/fulfillment_v1" / fixture).read_bytes()))


st.sidebar.button("Load synthetic source", on_click=load_fixture)
if page == "Order Fulfillment":
    data_fulfillment()
else:
    data_project()
