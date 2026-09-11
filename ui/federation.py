import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from federated.simulation import run_demo_simulation, get_simulation_summary, FederatedState
from federated.client import CLIENT_CONFIGS
from federated.strategy import get_strategy_info
from federated.server import get_server_info
from config import get_config
import pandas as pd


def _client_card(client_id: str, config: dict, status: str = "idle", loss: float = None, accuracy: float = None) -> None:
    status_colors = {
        "idle": "#555",
        "training": "#FF6B35",
        "completed": "#4CAF50",
        "error": "#f44336",
    }
    domain_emoji = {"Healthcare": "🏥", "Education": "📚", "Financial Literacy": "💰"}
    color = status_colors.get(status, "#555")
    emoji = domain_emoji.get(config["domain"], "📋")
    loss_str = f"Loss: {loss:.4f}" if loss is not None else "Loss: —"
    acc_str = f"Acc: {accuracy:.2%}" if accuracy is not None else "Acc: —"
    st.markdown(
        f"""
        <div style="background:#1e1e1e;border:1px solid {color};border-radius:8px;padding:14px;text-align:center">
            <div style="font-size:2em">{emoji}</div>
            <div style="color:#FF6B35;font-weight:bold;margin:4px 0">{client_id}</div>
            <div style="color:#ccc;font-size:0.85em">{config['domain']}</div>
            <div style="color:#aaa;font-size:0.8em">{config['language']}</div>
            <div style="margin-top:8px">
                <span style="background:{color};color:white;padding:2px 8px;border-radius:3px;font-size:0.75em">{status.upper()}</span>
            </div>
            <div style="color:#888;font-size:0.78em;margin-top:6px">{loss_str} · {acc_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_fedavg_diagram() -> None:
    fig = go.Figure()

    fig.add_shape(type="rect", x0=0.1, y0=0.7, x1=0.9, y1=0.95, line=dict(color="#FF6B35"), fillcolor="#2a1a0a")
    fig.add_annotation(x=0.5, y=0.825, text="Global Model (FedAvg Aggregation)", showarrow=False, font=dict(color="#FF6B35", size=12))

    positions = [(0.15, 0.35), (0.45, 0.35), (0.75, 0.35)]
    labels = ["Client 1\nHealthcare/Hindi", "Client 2\nEducation/Marathi", "Client 3\nFinancial/Tamil"]
    for (x, y), label in zip(positions, labels):
        fig.add_shape(type="rect", x0=x - 0.1, y0=y - 0.12, x1=x + 0.1, y1=y + 0.12, line=dict(color="#2196F3"), fillcolor="#0a1a2a")
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color="#2196F3", size=10))
        fig.add_annotation(x=x, y=(y + 0.12 + 0.7) / 2, text="Adapter\nWeights Only", showarrow=False, font=dict(color="#4CAF50", size=9))
        fig.add_shape(type="line", x0=x, y0=y + 0.12, x1=0.5, y1=0.7, line=dict(color="#4CAF50", dash="dash"))

    fig.add_shape(type="rect", x0=0.2, y0=0.05, x1=0.8, y1=0.18, line=dict(color="#888"), fillcolor="#111")
    fig.add_annotation(x=0.5, y=0.115, text="🔒 Raw Data stays local — Never transmitted", showarrow=False, font=dict(color="#aaa", size=11))

    fig.update_layout(
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <h2 style="color:#FF6B35">🌐 Federated Learning</h2>
        <p style="color:#aaa">
        Coordinate privacy-preserving federated training across multiple clients using FedAvg.
        Heavy computation runs on cloud GPU — this dashboard controls and visualises the process.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    strategy_info = get_strategy_info()

    st.markdown("### 📋 Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_rounds = st.number_input("Federated Rounds", min_value=1, max_value=3, value=cfg.federated_rounds)
    with col2:
        st.metric("Clients", cfg.num_clients)
    with col3:
        st.metric("Strategy", strategy_info["name"])
    with col4:
        st.metric("Environment", "Cloud GPU (Demo local)")

    st.warning(
        "⚠️ **Cloud Note:** Full QLoRA training with Flower requires a GPU environment. "
        "This dashboard runs a **[DEMO]** simulation locally to demonstrate the federated workflow. "
        "Use `cloud/run_federated.py` on a GPU machine for real training."
    )

    st.markdown("### 🏗️ Federated Architecture")
    _render_fedavg_diagram()

    st.markdown("### 👥 Clients")
    client_cols = st.columns(3)
    for idx, (client_id, config) in enumerate(CLIENT_CONFIGS.items()):
        with client_cols[idx]:
            if "fed_state" in st.session_state and st.session_state.fed_state.rounds_completed:
                last_round = st.session_state.fed_state.rounds_completed[-1]
                loss = last_round.client_losses.get(client_id)
                acc = last_round.client_accuracies.get(client_id)
                status = "completed" if st.session_state.fed_state.status == "completed" else "idle"
            else:
                loss = None
                acc = None
                status = "idle"
            _client_card(client_id, config, status, loss, acc)

    st.markdown("### 🚀 Run Simulation")
    if st.button("▶️ Run Demo Federated Simulation", type="primary"):
        with st.spinner(f"Running {num_rounds} federated round(s)... [DEMO]"):
            try:
                state = run_demo_simulation(num_rounds=num_rounds)
                st.session_state.fed_state = state
                st.success(f"✅ [DEMO] Simulation complete — {len(state.rounds_completed)} round(s)")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation error: {e}")

    if "fed_state" in st.session_state:
        state: FederatedState = st.session_state.fed_state
        summary = get_simulation_summary(state)

        st.markdown("### 📊 Round Results")

        if state.rounds_completed:
            rounds_data = []
            for r in state.rounds_completed:
                for client_id, loss in r.client_losses.items():
                    rounds_data.append({
                        "Round": r.round_num,
                        "Client": client_id,
                        "Loss": loss,
                        "Accuracy": r.client_accuracies.get(client_id, 0),
                    })

            df = pd.DataFrame(rounds_data)

            col_loss, col_acc = st.columns(2)
            with col_loss:
                fig_loss = px.line(df, x="Round", y="Loss", color="Client", title="[DEMO] Client Loss per Round",
                                   color_discrete_sequence=["#FF6B35", "#2196F3", "#4CAF50"])
                fig_loss.update_layout(paper_bgcolor="#111", plot_bgcolor="#1a1a1a", font_color="#ccc", height=260)
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_acc:
                fig_acc = px.line(df, x="Round", y="Accuracy", color="Client", title="[DEMO] Client Accuracy per Round",
                                  color_discrete_sequence=["#FF6B35", "#2196F3", "#4CAF50"])
                fig_acc.update_layout(paper_bgcolor="#111", plot_bgcolor="#1a1a1a", font_color="#ccc", height=260)
                st.plotly_chart(fig_acc, use_container_width=True)

            st.markdown("#### FedAvg Aggregation Flow")
            st.markdown(
                """
                ```
                Round N:
                  Client 1 (Healthcare/Hindi)   → local update Δw₁
                  Client 2 (Education/Marathi)  → local update Δw₂
                  Client 3 (Financial/Tamil)    → local update Δw₃
                           ↓ Clip + (optional) DP Noise
                  w_global = (Δw₁ + Δw₂ + Δw₃) / 3   [FedAvg]
                           ↓
                  Global adapter updated → sent back to clients
                ```
                """
            )

            total_time = sum(r.duration_seconds for r in state.rounds_completed)
            st.markdown(f"**Total simulation time:** {total_time:.3f}s · **Mode:** `[DEMO]`")
