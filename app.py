import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="LokLLM",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Global background & text ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #F7F7F7 !important;
        border-right: 2px solid #FF6B35 !important;
    }
    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }

    /* ── Sidebar radio buttons as nav links ── */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    [data-testid="stSidebar"] .stRadio label {
        display: block;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        font-size: 0.95em !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background 0.15s;
        color: #111111 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #FFE8DF !important;
        color: #FF6B35 !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
    [data-testid="stSidebar"] .stRadio input:checked ~ div {
        background-color: #FF6B35 !important;
        color: white !important;
    }
    /* Active radio item */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
        border-color: #FF6B35 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #FF6B35 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45em 1.4em !important;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background-color: #e55c2a !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #FF6B35 !important;
        border: 1.5px solid #FF6B35 !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextArea textarea {
        background-color: #FAFAFA !important;
        color: #111111 !important;
        border: 1.5px solid #E0E0E0 !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #FF6B35 !important;
        box-shadow: 0 0 0 2px rgba(255,107,53,0.15) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFF8F5;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #FF6B35;
    }
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 0.78em !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background-color: #FAFAFA !important;
        border: 1px solid #E8E8E8 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #111111 !important;
        font-weight: 600 !important;
    }

    /* ── Info / Success / Warning boxes ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        border: 1px solid #E8E8E8 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: #FF6B35 !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #FF6B35 !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        color: #111111 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #EBEBEB !important;
    }

    /* ── Code blocks ── */
    .stCodeBlock, code {
        background-color: #F5F5F5 !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #FF6B35 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FF6B35 !important;
        border-bottom-color: #FF6B35 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #FF6B35 !important;
    }

    /* hide default streamlit top right menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 16px 0;margin-bottom:8px">
            <div style="font-size:2.5em;line-height:1">🔒</div>
            <div style="color:#FF6B35;font-weight:800;font-size:1.15em;margin-top:6px;letter-spacing:0.01em">
                LokLLM
            </div>
            <div style="color:#888;font-size:0.72em;margin-top:2px">v0.1.0 · Hackathon Prototype</div>
        </div>
        <hr style="border:none;border-top:1.5px solid #FF6B35;margin:0 0 16px 0;opacity:0.35">
        <div style="color:#888;font-size:0.72em;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.08em;padding:0 6px;margin-bottom:8px">Navigation</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Home",
            "🎯  Personalization",
            "🌐  Federation",
            "🔒  Privacy Audit",
            "📈  Benchmarks",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <hr style="border:none;border-top:1px solid #E8E8E8;margin:20px 0 12px 0">
        <div style="color:#AAAAAA;font-size:0.70em;text-align:center;line-height:1.6">
            Privacy-Preserving On-Device<br>Personalization via Federated QLoRA<br><br>
            <span style="color:#FF6B35;font-weight:600">2026 Hackathon</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

try:
    if page == "🏠  Home":
        from ui.home import render
        render()
    elif page == "🎯  Personalization":
        from ui.personalization import render
        render()
    elif page == "🌐  Federation":
        from ui.federation import render
        render()
    elif page == "🔒  Privacy Audit":
        from ui.privacy import render
        render()
    elif page == "📈  Benchmarks":
        from ui.benchmarks import render
        render()
except Exception as exc:
    st.error(f"Page load error: {exc}")
    st.info("The application encountered an error on this page. Other pages remain accessible from the sidebar.")
    import traceback
    with st.expander("Error details"):
        st.code(traceback.format_exc())
