import streamlit as st
from data.synthetic_data import get_all_clients_data, CLIENT_PROFILES
from data.dataset import LocalDataset
from model.inference import generate_base_and_personalized
from core.hardware import get_execution_mode, EXECUTION_MODE_DEMO
from core.logger import get_logger

logger = get_logger(__name__)

_DOMAIN_EMOJI = {
    "Healthcare": "🏥",
    "Education": "📚",
    "Financial Literacy": "💰",
}


def _render_dataset_card(client_data: dict) -> None:
    domain = client_data["domain"]
    emoji = _DOMAIN_EMOJI.get(domain, "📋")
    st.markdown(
        f"""
        <div style="background:#1e1e1e;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:12px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <span style="font-size:1.5em">{emoji}</span>
                <div>
                    <span style="color:#FF6B35;font-weight:bold">{client_data['client_id']}</span>
                    <span style="color:#aaa;margin-left:8px">{domain} / {client_data['language']}</span>
                </div>
            </div>
            <div style="color:#ccc;font-size:0.9em">{client_data['description']}</div>
            <div style="color:#888;font-size:0.8em;margin-top:8px">{client_data['num_samples']} synthetic samples · Data stays on device</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        """
        <h2 style="color:#FF6B35">🎯 Personalization</h2>
        <p style="color:#aaa">
        Demonstrate local personalization using synthetic private data.
        Raw data never leaves the device — only adapter weights are transmitted.
        </p>
        """,
        unsafe_allow_html=True,
    )

    mode = get_execution_mode()

    st.markdown("### 📂 Synthetic Local Datasets")
    st.info(
        "These synthetic datasets represent private local user data. "
        "**They are never transmitted.** Only the resulting adapter weights leave the device."
    )

    try:
        all_data = get_all_clients_data()
    except Exception as e:
        st.error(f"Failed to load synthetic data: {e}")
        return

    selected_client_id = st.selectbox(
        "Select client profile",
        options=[d["client_id"] for d in all_data],
        format_func=lambda cid: next(
            f"{d['client_id']} — {d['domain']} / {d['language']}"
            for d in all_data if d["client_id"] == cid
        ),
    )

    selected_data = next(d for d in all_data if d["client_id"] == selected_client_id)

    col_cards = st.columns(len(all_data))
    for idx, client_data in enumerate(all_data):
        with col_cards[idx]:
            _render_dataset_card(client_data)

    st.markdown("### 🔍 Sample Data")
    dataset = LocalDataset.from_dict(selected_data)
    sample_count = st.slider("Samples to preview", min_value=1, max_value=min(5, len(dataset)), value=3)

    sample_df_data = []
    for i, sample in enumerate(dataset.samples[:sample_count]):
        sample_df_data.append({
            "Input (Local Language)": sample["input"],
            "Expected Response": sample["output"],
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(sample_df_data), use_container_width=True)

    st.markdown("### 💬 Inference Demo")
    st.markdown(
        f"""
        <div style="background:#2a1a0a;border:1px solid #FF6B35;border-radius:6px;padding:10px;margin-bottom:12px">
            <strong style="color:#FF6B35">Mode:</strong>
            <span style="color:#ccc"> {mode} — {'Responses are simulated' if mode == EXECUTION_MODE_DEMO else 'Using local model'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_prompts = dataset.get_prompts()
    prompt_options = ["(Type your own)"] + sample_prompts[:5]
    chosen = st.selectbox("Choose a sample prompt or type your own", prompt_options)

    if chosen == "(Type your own)":
        prompt = st.text_input("Enter your prompt", value="")
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
                <div style="background:#1e1e1e;border:1px solid #444;border-radius:6px;padding:14px">
                    <div style="color:#888;font-size:0.75em;margin-bottom:8px">
                        Mode: {base['mode']} · Latency: {base['latency_ms']} ms
                    </div>
                    <div style="color:#eee">{base['response']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_pers:
            st.markdown("#### ✨ Personalized Response")
            pers = results["personalized_response"]
            st.markdown(
                f"""
                <div style="background:#1a1e1a;border:1px solid #4CAF50;border-radius:6px;padding:14px">
                    <div style="color:#888;font-size:0.75em;margin-bottom:8px">
                        Mode: {pers['mode']} · Personalization: {'Applied' if results['personalization_applied'] else 'None'}
                    </div>
                    <div style="color:#eee">{pers['response']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🔒 Privacy Reminder")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.success("✅ Raw data: **0 bytes transmitted**")
    with col_p2:
        st.success("✅ Training: **local only**")
    with col_p3:
        st.success("✅ Adapter weights: **transmitted (not raw data)**")
