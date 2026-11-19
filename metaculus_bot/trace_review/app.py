"""The trace-review app: ``make review`` (streamlit run metaculus_bot/trace_review/app.py).

Reads the bundle ``make trace_index`` wrote (or ``$TRACE_REVIEW_BUNDLE_DIR``); opens no network
connection and spends nothing.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Trace review", layout="wide", initial_sidebar_state="expanded")
st.navigation(
    [
        st.Page("pages/forecasts.py", title="Forecasts", default=True),
        st.Page("pages/detail.py", title="Forecast detail", url_path="detail"),
        st.Page("pages/models.py", title="Models"),
    ]
).run()
