"""Restrained visual defaults shared by the existing Streamlit pages."""

import streamlit as st


def apply_styles():
    st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] { border-right: 1px solid #dce3ed; }
    h1 { font-size: 2rem !important; letter-spacing: -.035em; }
    h2 { font-size: 1.4rem !important; letter-spacing: -.02em; }
    h3 { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stTabs"] { margin-top: .5rem; }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 2rem; }
        h1 { font-size: 1.75rem !important; }
    }
</style>
""", unsafe_allow_html=True)
