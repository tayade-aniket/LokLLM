import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="PRIVFEDQLORA",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #111111;
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #e55c2a;
    }
    .stSelectbox > div > div {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stMetric label {
        color: #aaa !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #fff !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0;border-bottom:1px solid #333;margin-bottom:16px">
            <div style="font-size:2em">🔒</div>
            <div style="color:#FF6B35;font-weight:bold;font-size:1.1em">PRIVFEDQLORA</div>
            <div style="color:#888;font-size:0.75em">v0.1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.selectbox(
        "Navigate",
        options=["🏠 Home", "🎯 Personalization", "🌐 Federation", "🔒 Privacy Audit", "📈 Benchmarks"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="margin-top:auto;padding-top:20px;border-top:1px solid #333;color:#555;font-size:0.72em;text-align:center">
        Privacy-Preserving On-Device Personalization<br>via Federated QLoRA<br><br>
        Hackathon Prototype — 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    if page == "🏠 Home":
        from ui.home import render
        render()
    elif page == "🎯 Personalization":
        from ui.personalization import render
        render()
    elif page == "🌐 Federation":
        from ui.federation import render
        render()
    elif page == "🔒 Privacy Audit":
        from ui.privacy import render
        render()
    elif page == "📈 Benchmarks":
        from ui.benchmarks import render
        render()
except Exception as exc:
    st.error(f"Page load error: {exc}")
    st.info("The application encountered an error loading this page. Core functionality may still be available from other pages.")
    import traceback
    with st.expander("Error details"):
        st.code(traceback.format_exc())
