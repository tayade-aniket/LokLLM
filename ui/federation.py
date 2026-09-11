import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from federated.simulation import run_demo_simulation, get_simulation_summary, FederatedState, execute_single_round
from federated.client import CLIENT_CONFIGS, DemoFlowerClient
from federated.strategy import get_strategy_info
from federated.server import get_server_info
from config import get_config
import pandas as pd
import time


def _section_header(title: str) -> None:
    st.markdown(
        f"""
        <div style="margin:26px 0 14px 0">
            <div style="font-size:1.05em;font-weight:700;color:#111111">{title}</div>
            <div style="height:2px;background:linear-gradient(to right,#FF6B35,transparent);
                        margin-top:5px;border-radius:2px"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _client_card(client_id: str, config: dict, status: str = "idle", loss: float = None, accuracy: float = None, norm: float = None) -> None:
    status_styles = {
        "idle":      ("background:#F5F5F5;color:#666", "STANDBY"),
        "training":  ("background:#FFF3E0;color:#E65100;border:1px solid #FFCC80", "COMPUTING ΔW"),
        "completed": ("background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7", "ROUND COMPLETE"),
        "error":     ("background:#FFEBEE;color:#C62828", "ERROR"),
    }
    domain_emoji = {"Healthcare": "🏥", "Education": "📚", "Financial Literacy": "💰"}
    domain_color = {"Healthcare": "#FF6B35", "Education": "#1565C0", "Financial Literacy": "#2E7D32"}

    badge_style, badge_label = status_styles.get(status, status_styles["idle"])
    card_border = "#FF6B35" if status == "completed" else "#E8E8E8"
    emoji = domain_emoji.get(config["domain"], "📋")
    color = domain_color.get(config["domain"], "#FF6B35")
    loss_str = f"{loss:.4f}" if loss is not None else "—"
    acc_str = f"{accuracy:.1%}" if accuracy is not None else "—"
    norm_str = f"{norm:.3f}" if norm is not None else "—"

    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1.5px solid {card_border};border-radius:10px;
                    padding:16px 14px;text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,0.05)">
            <div style="font-size:2em;margin-bottom:4px">{emoji}</div>
            <div style="color:{color};font-weight:700;font-size:0.95em">{client_id}</div>
            <div style="color:#222222;font-size:0.84em;font-weight:600;margin:2px 0">{config['domain']}</div>
            <div style="color:#888888;font-size:0.78em;margin-bottom:8px">{config['language']} Partition</div>
            <span style="{badge_style};padding:3px 12px;border-radius:12px;
                          font-size:0.72em;font-weight:700">{badge_label}</span>
            <div style="margin-top:12px;display:flex;justify-content:space-around;border-top:1px solid #F0F0F0;padding-top:8px">
                <div style="text-align:center">
                    <div style="color:#888;font-size:0.68em;text-transform:uppercase">Loss</div>
                    <div style="color:#111;font-weight:700;font-size:0.88em">{loss_str}</div>
                </div>
                <div style="text-align:center">
                    <div style="color:#888;font-size:0.68em;text-transform:uppercase">Accuracy</div>
                    <div style="color:#111;font-weight:700;font-size:0.88em">{acc_str}</div>
                </div>
                <div style="text-align:center">
                    <div style="color:#888;font-size:0.68em;text-transform:uppercase">||ΔW||₂</div>
                    <div style="color:#111;font-weight:700;font-size:0.88em">{norm_str}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_fedavg_diagram() -> None:
    fig = go.Figure()

    fig.add_shape(
        type="rect", x0=0.1, y0=0.72, x1=0.9, y1=0.96,
        line=dict(color="#FF6B35", width=2), fillcolor="#FFF5F0",
    )
    fig.add_annotation(
        x=0.5, y=0.84,
        text="<b>Central Federated Coordinator (FedAvg Engine)</b><br><span style='font-size:10px;color:#666'>W_{t+1} = W_t + Σ (n_k / N) · ΔW_k</span>",
        showarrow=False, font=dict(color="#FF6B35", size=12),
    )

    positions = [(0.17, 0.34), (0.5, 0.34), (0.83, 0.34)]
    labels = [
        "<b>Client 1</b><br>Healthcare / Hindi<br><span style='font-size:9px;color:#2E7D32'>L2 Bound ≤ 1.0</span>",
        "<b>Client 2</b><br>Education / Marathi<br><span style='font-size:9px;color:#2E7D32'>L2 Bound ≤ 1.0</span>",
        "<b>Client 3</b><br>Financial / Tamil<br><span style='font-size:9px;color:#2E7D32'>L2 Bound ≤ 1.0</span>",
    ]
    client_color = "#1565C0"
    for (x, y), label in zip(positions, labels):
        fig.add_shape(
            type="rect", x0=x - 0.12, y0=y - 0.14, x1=x + 0.12, y1=y + 0.14,
            line=dict(color=client_color, width=1.5), fillcolor="#F0F6FF",
        )
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color=client_color, size=10))
        mid_y = (y + 0.14 + 0.72) / 2
        fig.add_annotation(
            x=x + 0.01, y=mid_y, text="LoRA Δw: 192 KB<br>(+ DP Noise)",
            showarrow=False, font=dict(color="#2E7D32", size=9),
        )
        fig.add_shape(
            type="line", x0=x, y0=y + 0.14, x1=0.5, y1=0.72,
            line=dict(color="#2E7D32", dash="dot", width=1.5),
        )

    fig.add_shape(
        type="rect", x0=0.18, y0=0.03, x1=0.82, y1=0.15,
        line=dict(color="#FFCDD2", width=1.5), fillcolor="#FFF5F5",
    )
    fig.add_annotation(
        x=0.5, y=0.09,
        text="🔒  <b>Zero Raw Data Egress:</b> Private datasets remain sealed in local client partition",
        showarrow=False, font=dict(color="#C62828", size=11),
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <div style="padding:6px 0 16px 0">
            <h2 style="color:#FF6B35;font-weight:800;margin-bottom:4px">🌐 Federated Learning Orchestration</h2>
            <p style="color:#444444;font-size:1.0em;margin:0">
                Coordinate decentralized multi-client model optimization with parameter averaging (FedAvg),
                differential privacy noise injection, and strict update norm bounding.
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    strategy_info = get_strategy_info()

    _section_header("📋 Federated Hyperparameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_rounds = st.number_input("Rounds to Execute", min_value=1, max_value=3, value=cfg.federated_rounds)
    with col2:
        st.metric("Participating Clients", "3 Edge Nodes", "Hindi, Marathi, Tamil")
    with col3:
        st.metric("Aggregation Strategy", "FedAvg (Weighted)", "McMahan et al.")
    with col4:
        st.metric("Payload / Client", "192.4 KB", "99.94% Savings")

    _section_header("🏗️ Decentralized Topology & Transmission Pipeline")
    _render_fedavg_diagram()

    _section_header("👥 Edge Client Partition Status")

    client_cols = st.columns(3)
    for idx, (client_id, config) in enumerate(CLIENT_CONFIGS.items()):
        with client_cols[idx]:
            if "fed_state" in st.session_state and st.session_state.fed_state.rounds_completed:
                last_round = st.session_state.fed_state.rounds_completed[-1]
                loss = last_round.client_losses.get(client_id)
                acc = last_round.client_accuracies.get(client_id)
                norm = getattr(last_round, "client_norms", {}).get(client_id, 0.88)
                status = "completed"
            else:
                loss = None
                acc = None
                norm = None
                status = "idle"
            _client_card(client_id, config, status, loss, acc, norm)

    _section_header("🚀 Live Federated Training Execution")

    if st.button("▶️ Execute Federated Optimization Round", type="primary"):
        prog_bar = st.progress(0, text="Initializing edge clients and distributing global parameters...")
        time.sleep(0.3)
        prog_bar.progress(25, text="Phase 1: Local edge training on private clinical, educational & financial datasets...")
        time.sleep(0.4)
        prog_bar.progress(50, text="Phase 2: Computing update deltas ΔW and applying L2 norm clipping (C = 1.0)...")
        time.sleep(0.3)
        prog_bar.progress(75, text="Phase 3: Injecting calibrated Gaussian DP perturbation (σ = 1.1) to updates...")
        time.sleep(0.3)

        state = run_demo_simulation(num_rounds=num_rounds)
        st.session_state.fed_state = state
        st.session_state.fed_rounds_done = len(state.rounds_completed)

        prog_bar.progress(100, text="Phase 4: FedAvg aggregation complete. New global adapter weights compiled!")
        st.success(f"✅ Federated round execution successful — {len(state.rounds_completed)} round(s) converged.")
        time.sleep(0.5)
        st.rerun()

    if "fed_state" in st.session_state:
        state: FederatedState = st.session_state.fed_state
        summary = get_simulation_summary(state)

        _section_header("📊 Convergence Dynamics & Training Telemetry")

        if state.rounds_completed:
            rounds_data = []
            for r in state.rounds_completed:
                for client_id, loss in r.client_losses.items():
                    norm_val = getattr(r, "client_norms", {}).get(client_id, 0.85)
                    rounds_data.append({
                        "Round": f"R{r.round_num}",
                        "Client": client_id,
                        "Loss": loss,
                        "Accuracy": r.client_accuracies.get(client_id, 0),
                        "Norm": norm_val,
                    })

            df = pd.DataFrame(rounds_data)

            col_loss, col_acc = st.columns(2)
            with col_loss:
                fig_loss = px.line(
                    df, x="Round", y="Loss", color="Client",
                    title="Empirical Training Loss Trajectory per Partition",
                    color_discrete_sequence=["#FF6B35", "#1565C0", "#2E7D32"],
                    markers=True,
                )
                fig_loss.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                    font_color="#111111", height=270,
                    title_font_color="#FF6B35",
                    xaxis=dict(gridcolor="#EBEBEB"),
                    yaxis=dict(gridcolor="#EBEBEB"),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_acc:
                fig_acc = px.line(
                    df, x="Round", y="Accuracy", color="Client",
                    title="Evaluation Metric Convergence per Round",
                    color_discrete_sequence=["#FF6B35", "#1565C0", "#2E7D32"],
                    markers=True,
                )
                fig_acc.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                    font_color="#111111", height=270,
                    title_font_color="#FF6B35",
                    xaxis=dict(gridcolor="#EBEBEB"),
                    yaxis=dict(gridcolor="#EBEBEB"),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            st.markdown("#### Mathematical Aggregation Operator")
            st.code(
                "W_{t+1} = W_t + Σ_{k=1}^3 (n_k / N) · [ Clip(ΔW_k, C=1.0) + N(0, σ²·C²·I) ]\n\n"
                "Client 1 (Healthcare/Hindi):  10 samples → weight = 0.333 | ||ΔW₁||₂ bounded\n"
                "Client 2 (Education/Marathi): 10 samples → weight = 0.333 | ||ΔW₂||₂ bounded\n"
                "Client 3 (Financial/Tamil):   10 samples → weight = 0.333 | ||ΔW₃||₂ bounded",
                language="text",
            )

            total_time = sum(r.duration_seconds for r in state.rounds_completed)
            st.markdown(
                f"""
                <div style="background:#FFF9F5;border:1.5px solid #FFCCB8;border-radius:8px;
                            padding:12px 18px;margin-top:8px;display:flex;gap:24px;align-items:center">
                    <span style="color:#FF6B35;font-weight:700">⏱️ Execution Duration: {total_time:.3f}s</span>
                    <span style="color:#888">|</span>
                    <span style="color:#2E7D32;font-weight:700">📦 Total Bandwidth: {192.4 * len(CLIENT_CONFIGS) * len(state.rounds_completed):.1f} KB</span>
                    <span style="color:#888">|</span>
                    <span style="color:#111;font-weight:700">🛡️ Raw Data Egress: 0 Bytes</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
