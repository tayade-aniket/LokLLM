import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from federated.simulation import run_demo_simulation, get_simulation_summary, FederatedState
from federated.client import CLIENT_CONFIGS
from federated.strategy import get_strategy_info
from federated.server import get_server_info
from config import get_config
import pandas as pd


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


def _client_card(client_id: str, config: dict, status: str = "idle", loss: float = None, accuracy: float = None) -> None:
    status_styles = {
        "idle":      ("background:#F5F5F5;color:#666", "IDLE"),
        "training":  ("background:#FF6B35;color:white", "TRAINING"),
        "completed": ("background:#2E7D32;color:white", "DONE"),
        "error":     ("background:#C62828;color:white", "ERROR"),
    }
    domain_emoji = {"Healthcare": "🏥", "Education": "📚", "Financial Literacy": "💰"}
    domain_color = {"Healthcare": "#FF6B35", "Education": "#1565C0", "Financial Literacy": "#2E7D32"}

    badge_style, badge_label = status_styles.get(status, status_styles["idle"])
    card_border = "#FF6B35" if status == "completed" else "#E8E8E8"
    emoji = domain_emoji.get(config["domain"], "📋")
    color = domain_color.get(config["domain"], "#FF6B35")
    loss_str = f"{loss:.4f}" if loss is not None else "—"
    acc_str = f"{accuracy:.1%}" if accuracy is not None else "—"

    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1.5px solid {card_border};border-radius:10px;
                    padding:18px 14px;text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,0.07)">
            <div style="font-size:2em;margin-bottom:6px">{emoji}</div>
            <div style="color:{color};font-weight:700;font-size:0.95em">{client_id}</div>
            <div style="color:#333333;font-size:0.82em;margin:3px 0">{config['domain']}</div>
            <div style="color:#888888;font-size:0.78em;margin-bottom:10px">{config['language']}</div>
            <span style="{badge_style};padding:3px 12px;border-radius:12px;
                          font-size:0.72em;font-weight:700">{badge_label}</span>
            <div style="margin-top:10px;display:flex;justify-content:center;gap:12px">
                <div style="text-align:center">
                    <div style="color:#888;font-size:0.68em;text-transform:uppercase">Loss</div>
                    <div style="color:#111;font-weight:700;font-size:0.9em">{loss_str}</div>
                </div>
                <div style="text-align:center">
                    <div style="color:#888;font-size:0.68em;text-transform:uppercase">Acc</div>
                    <div style="color:#111;font-weight:700;font-size:0.9em">{acc_str}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_fedavg_diagram() -> None:
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=0.1, y0=0.7, x1=0.9, y1=0.95,
        line=dict(color="#FF6B35", width=2), fillcolor="#FFF0EA",
    )
    fig.add_annotation(
        x=0.5, y=0.825,
        text="<b>Global Model — FedAvg Aggregation</b>",
        showarrow=False, font=dict(color="#FF6B35", size=13),
    )

    positions = [(0.17, 0.36), (0.5, 0.36), (0.83, 0.36)]
    labels = ["Client 1<br>Healthcare / Hindi", "Client 2<br>Education / Marathi", "Client 3<br>Financial / Tamil"]
    client_color = "#1565C0"
    for (x, y), label in zip(positions, labels):
        fig.add_shape(
            type="rect", x0=x - 0.11, y0=y - 0.13, x1=x + 0.11, y1=y + 0.13,
            line=dict(color=client_color, width=1.5), fillcolor="#EEF4FF",
        )
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color=client_color, size=10))
        mid_y = (y + 0.13 + 0.7) / 2
        fig.add_annotation(
            x=x + 0.01, y=mid_y, text="Adapter Δw only",
            showarrow=False, font=dict(color="#2E7D32", size=9),
        )
        fig.add_shape(
            type="line", x0=x, y0=y + 0.13, x1=0.5, y1=0.7,
            line=dict(color="#2E7D32", dash="dot", width=1.5),
        )

    fig.add_shape(
        type="rect", x0=0.2, y0=0.03, x1=0.8, y1=0.16,
        line=dict(color="#E0E0E0"), fillcolor="#F5F5F5",
    )
    fig.add_annotation(
        x=0.5, y=0.095,
        text="🔒  Raw Data stays on device — never transmitted",
        showarrow=False, font=dict(color="#C62828", size=11),
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <div style="padding:8px 0 20px 0">
            <h2 style="color:#FF6B35;font-weight:800;margin-bottom:4px">🌐 Federated Learning</h2>
            <p style="color:#555555;font-size:1.0em;margin:0">
                Coordinate privacy-preserving federated training across multiple clients using FedAvg.
                Heavy computation runs on cloud GPU — this dashboard controls and visualises the process.
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    strategy_info = get_strategy_info()

    _section_header("📋 Configuration")

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

    _section_header("🏗️ Federated Architecture")
    _render_fedavg_diagram()

    _section_header("👥 Clients")

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

    _section_header("🚀 Run Simulation")

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

        _section_header("📊 Round Results")

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
                fig_loss = px.line(
                    df, x="Round", y="Loss", color="Client",
                    title="[DEMO] Client Loss per Round",
                    color_discrete_sequence=["#FF6B35", "#1565C0", "#2E7D32"],
                )
                fig_loss.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                    font_color="#111111", height=260,
                    title_font_color="#FF6B35",
                    xaxis=dict(gridcolor="#E8E8E8"),
                    yaxis=dict(gridcolor="#E8E8E8"),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_acc:
                fig_acc = px.line(
                    df, x="Round", y="Accuracy", color="Client",
                    title="[DEMO] Client Accuracy per Round",
                    color_discrete_sequence=["#FF6B35", "#1565C0", "#2E7D32"],
                )
                fig_acc.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                    font_color="#111111", height=260,
                    title_font_color="#FF6B35",
                    xaxis=dict(gridcolor="#E8E8E8"),
                    yaxis=dict(gridcolor="#E8E8E8"),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            st.markdown("#### FedAvg Aggregation Flow")
            st.code(
                "Round N:\n"
                "  Client 1 (Healthcare/Hindi)   → local update Δw₁\n"
                "  Client 2 (Education/Marathi)  → local update Δw₂\n"
                "  Client 3 (Financial/Tamil)    → local update Δw₃\n"
                "           ↓ Clip + (optional) DP Noise\n"
                "  w_global = (Δw₁ + Δw₂ + Δw₃) / 3   [FedAvg]\n"
                "           ↓\n"
                "  Global adapter updated → distributed back to clients",
                language="text",
            )

            total_time = sum(r.duration_seconds for r in state.rounds_completed)
            st.markdown(
                f"""
                <div style="background:#FFF8F5;border:1px solid #FFD5C2;border-radius:8px;
                            padding:10px 16px;margin-top:8px;display:flex;gap:20px">
                    <span style="color:#FF6B35;font-weight:700">Total time: {total_time:.3f}s</span>
                    <span style="color:#888">·</span>
                    <span style="color:#888">Mode: <code>[DEMO]</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
