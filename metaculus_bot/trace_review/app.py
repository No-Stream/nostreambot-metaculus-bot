"""The trace-review app: ``make review`` (streamlit run metaculus_bot/trace_review/app.py).

Reads the bundle ``make trace_index`` wrote (or ``$TRACE_REVIEW_BUNDLE_DIR``); opens no network
connection and spends nothing.
"""

from __future__ import annotations

import streamlit as st

# The page files are not in a folder named pages/: with one, Streamlit runs a page file directly when
# its URL is opened (a pasted /detail link), skipping this file, its navigation and its wide layout.
st.set_page_config(page_title="Trace review", layout="wide", initial_sidebar_state="expanded")
st.navigation(
    [
        st.Page("app_pages/forecasts.py", title="Forecasts", default=True),
        st.Page("app_pages/detail.py", title="Forecast detail", url_path="detail"),
        st.Page("app_pages/models.py", title="Models"),
    ]
).run()
