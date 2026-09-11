import streamlit as st
from data.synthetic_data import get_all_clients_data, CLIENT_PROFILES
from data.dataset import LocalDataset
from model.inference import generate_base_and_personalized
from core.hardware import get_execution_mode, EXECUTION_MODE_DEMO
from core.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

_DOMAIN_EMOJI = {
    "Healthcare": "🏥",
    "Education": "📚",
    "Financial Literacy": "💰",
}

_DOMAIN_COLOR = {
    "Healthcare": "#FF6B35",
    "Education": "#1565C0",
    "Financial Literacy": "#2E7D32",
}


def _section_header(title: str) -> None:
    st.markdown(
        f"""
        <div style="margin:28px 0 14px 0">
            <div style="font-size:1.05em;font-weight:700;color:#111111">{title}</div>
            <div style="height:2px;background:linear-gradient(to right,#FF6B35,transparent);
                        margin-top:5px;border-radius:2px"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_card(client_data: dict, selected: bool = False) -> None:
    domain = client_data["domain"]
    emoji = _DOMAIN_EMOJI.get(domain, "📋")
    color = _DOMAIN_COLOR.get(domain, "#FF6B35")
    border = f"2px solid {color}" if selected else "1px solid #E8E8E8"
    bg = "#FFF8F5" if selected else "#FFFFFF"
    st.markdown(
        f"""
        <div style="background:{bg};border:{border};border-radius:10px;
                    padding:16px;margin-bottom:8px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="font-size:1.6em">{emoji}</span>
                <div>
                    <div style="color:{color};font-weight:700;font-size:0.95em">
                        {client_data['client_id']}
                    </div>
                    <div style="color:#555555;font-size:0.8em">{domain} · {client_data['language']}</div>
                </div>
            </div>
            <div style="color:#444444;font-size:0.85em;line-height:1.5">
                {client_data['description']}
            </div>
            <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
                <span style="background:#F0F0F0;color:#555;font-size:0.75em;
                             padding:3px 10px;border-radius:12px">
                    {client_data['num_samples']} samples
                </span>
                <span style="background:#E8F5E9;color:#2E7D32;font-size:0.75em;
                             padding:3px 10px;border-radius:12px">
                    🔒 stays on device
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        """
        <div style="padding:8px 0 20px 0">
            <h2 style="color:#FF6B35;font-weight:800;margin-bottom:4px">🎯 Personalization</h2>
            <p style="color:#555555;font-size:1.0em;margin:0">
                Demonstrate local personalization using synthetic private data.
                Raw data never leaves the device — only adapter weights are transmitted.
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
        """,
        unsafe_allow_html=True,
    )

    mode = get_execution_mode()

    st.info(
        "🔒 **Privacy guarantee:** These synthetic datasets represent private local user data. "
        "**They are never transmitted.** Only the resulting LoRA adapter weights (~192 KB) leave the device."
    )

    try:
        all_data = get_all_clients_data()
    except Exception as e:
        st.error(f"Failed to load synthetic data: {e}")
        return

    _section_header("📂 Client Profiles")

    selected_client_id = st.selectbox(
        "Active client",
        options=[d["client_id"] for d in all_data],
        format_func=lambda cid: next(
            f"{_DOMAIN_EMOJI.get(d['domain'], '📋')}  {d['client_id']} — {d['domain']} / {d['language']}"
            for d in all_data if d["client_id"] == cid
        ),
    )

    selected_data = next(d for d in all_data if d["client_id"] == selected_client_id)

    col_cards = st.columns(len(all_data))
    for idx, client_data in enumerate(all_data):
        with col_cards[idx]:
            _render_dataset_card(client_data, selected=client_data["client_id"] == selected_client_id)

    _section_header("🔍 Sample Data Preview")

    dataset = LocalDataset.from_dict(selected_data)
    sample_count = st.slider("Samples to preview", min_value=1, max_value=min(5, len(dataset)), value=3)

    sample_df_data = []
    for sample in dataset.samples[:sample_count]:
        sample_df_data.append({
            "Input (Local Language)": sample["input"],
            "Expected Response": sample["output"],
        })

    st.dataframe(pd.DataFrame(sample_df_data), use_container_width=True)

    _section_header("💬 Inference Demo")

    mode_color = "#FF6B35" if mode == EXECUTION_MODE_DEMO else "#2E7D32"
    mode_label = "Responses are simulated — DEMO mode" if mode == EXECUTION_MODE_DEMO else "Using local model"
    st.markdown(
        f"""
        <div style="background:#FFF8F5;border:1px solid #FFD5C2;border-left:4px solid {mode_color};
                    border-radius:8px;padding:10px 16px;margin-bottom:16px;
                    display:flex;align-items:center;gap:10px">
            <span style="background:{mode_color};color:white;padding:2px 10px;border-radius:12px;
                          font-size:0.78em;font-weight:700">{mode}</span>
            <span style="color:#444444;font-size:0.88em">{mode_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_prompts = dataset.get_prompts()
    prompt_options = ["(Type your own)"] + sample_prompts[:5]
    chosen = st.selectbox("Choose a sample prompt", prompt_options)

    if chosen == "(Type your own)":
        prompt = st.text_input("Enter your prompt", value="", placeholder="Type a question...")
    else:
        prompt = st.text_input("Prompt", value=chosen)

    if st.button("🚀 Generate Response", type="primary") and prompt.strip():
        with st.spinner("Generating response..."):
            try:
                results = generate_base_and_personalized(
                    prompt=prompt,
                    model=None,
                    tokenizer=None,
                    personalization_context=selected_data["description"],
                    domain=selected_data["domain"].lower().replace(" ", "_"),
                )
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        col_base, col_pers = st.columns(2)

        with col_base:
            st.markdown("#### 🤖 Base Response")
            base = results["base_response"]
            st.markdown(
                f"""
                <div style="background:#FAFAFA;border:1px solid #E0E0E0;border-radius:8px;padding:16px">
                    <div style="color:#888888;font-size:0.75em;margin-bottom:10px;
                                display:flex;gap:8px;align-items:center">
                        <span style="background:#E8E8E8;color:#555;padding:2px 8px;
                                     border-radius:10px;font-size:0.9em">{base['mode']}</span>
                        <span>Latency: {base['latency_ms']} ms</span>
                    </div>
                    <div style="color:#222222;line-height:1.65;font-size:0.92em">{base['response']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_pers:
            st.markdown("#### ✨ Personalized Response")
            pers = results["personalized_response"]
            personalization_label = "Applied" if results["personalization_applied"] else "None"
            st.markdown(
                f"""
                <div style="background:#F0FFF4;border:1px solid #A5D6A7;border-radius:8px;padding:16px">
                    <div style="color:#888888;font-size:0.75em;margin-bottom:10px;
                                display:flex;gap:8px;align-items:center">
                        <span style="background:#C8E6C9;color:#2E7D32;padding:2px 8px;
                                     border-radius:10px;font-size:0.9em">{pers['mode']}</span>
                        <span>Personalization: {personalization_label}</span>
                    </div>
                    <div style="color:#222222;line-height:1.65;font-size:0.92em">{pers['response']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _section_header("🔒 Privacy Status")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.success("✅ Raw data — **0 bytes transmitted**")
    with col_p2:
        st.success("✅ Training — **local only**")
    with col_p3:
        st.success("✅ Adapter weights — **only output transmitted**")
