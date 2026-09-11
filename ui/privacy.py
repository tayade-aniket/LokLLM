import streamlit as st
import plotly.graph_objects as go
from privacy.audit import build_audit_report, get_privacy_boundary_summary
from privacy.clipping import get_clipping_info
from privacy.differential_privacy import get_dp_info
from privacy.secure_aggregation import get_secure_agg_info
from config import get_config


def _status_badge(status: str) -> str:
    colors = {
        "IMPLEMENTED": "#4CAF50",
        "DEMONSTRATION": "#FF6B35",
        "SIMULATION": "#2196F3",
        "NOT IMPLEMENTED": "#f44336",
        "YES": "#4CAF50",
        "NO": "#4CAF50",
        "Enabled": "#4CAF50",
        "Disabled": "#888",
        "Simulation": "#2196F3",
    }
    color = colors.get(status, "#888")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:4px;font-size:0.8em;font-weight:bold">{status}</span>'


def _privacy_metric_row(label: str, value: str, badge: str = None) -> None:
    badge_html = _status_badge(badge) if badge else ""
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:10px 16px;border-bottom:1px solid #222;background:#1a1a1a">
            <span style="color:#ccc;font-size:0.95em">{label}</span>
            <span style="color:#fff;font-weight:bold">{value} {badge_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_flow_diagram() -> None:
    fig = go.Figure()

    nodes = [
        (0.1, 0.5, "🔐 Raw\nPersonal Data", "#f44336", "#2a0a0a"),
        (0.35, 0.5, "📱 Local\nDevice", "#FF6B35", "#2a1a0a"),
        (0.6, 0.5, "✂️ Clip +\nDP Noise", "#FF9800", "#2a1a08"),
        (0.85, 0.5, "☁️ Federated\nServer", "#2196F3", "#0a1a2a"),
    ]

    for x, y, label, color, bg in nodes:
        fig.add_shape(type="circle", x0=x - 0.08, y0=y - 0.18, x1=x + 0.08, y1=y + 0.18,
                      line=dict(color=color, width=2), fillcolor=bg)
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color=color, size=10))

    arrows = [
        (0.18, 0.5, 0.27, 0.5, "#4CAF50", "Local only\n(no upload)"),
        (0.43, 0.5, 0.52, 0.5, "#FF6B35", "Adapter Δw\n(not raw data)"),
        (0.68, 0.5, 0.77, 0.5, "#2196F3", "Noisy Δw"),
    ]
    for x0, y0, x1, y1, color, label in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y",
                           arrowhead=2, arrowsize=1.5, arrowcolor=color,
                           text=label, font=dict(color=color, size=9),
                           xanchor="center", yanchor="bottom", yshift=20)

    fig.add_shape(type="rect", x0=0.0, y0=0.15, x1=0.22, y1=0.85,
                  line=dict(color="#f44336", dash="dot"), fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=0.11, y=0.1, text="🚫 Never\nleaves device", showarrow=False, font=dict(color="#f44336", size=9))

    fig.update_layout(
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <h2 style="color:#FF6B35">🔒 Privacy Audit</h2>
        <p style="color:#aaa">
        Transparency report on what data is transmitted, what stays local, and which privacy mechanisms are active.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    report = build_audit_report(mode="DEMO")
    boundary = get_privacy_boundary_summary()
    clipping_info = get_clipping_info(cfg.dp_max_grad_norm)
    dp_info = get_dp_info(cfg)
    sec_agg_info = get_secure_agg_info(cfg.secure_agg_enabled)

    st.markdown("### 🔍 Privacy Boundary — What Gets Transmitted")
    _render_data_flow_diagram()

    st.markdown("### 📋 Audit Report")
    st.markdown('<div style="background:#111;border:1px solid #333;border-radius:8px;overflow:hidden">', unsafe_allow_html=True)
    _privacy_metric_row("Raw personal data uploaded", "0 bytes", "NO")
    _privacy_metric_row("Raw dataset transmitted", "NO", "NO")
    _privacy_metric_row("Local training performed", "YES (cloud simulation)", "YES")
    _privacy_metric_row("Adapter weights transmitted", "YES — LoRA adapter only", "YES")
    _privacy_metric_row("Update clipping", f"L2 norm ≤ {cfg.dp_max_grad_norm}", clipping_info["status"])
    _privacy_metric_row("Differential privacy", f"ε={cfg.dp_epsilon}, δ={cfg.dp_delta}", dp_info["status"])
    _privacy_metric_row("Secure aggregation", sec_agg_info["description"][:40] + "…", sec_agg_info["status"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🛡️ Mechanism Details")

    with st.expander("✂️ Update Clipping — IMPLEMENTED"):
        st.markdown(f"""
        - **Status:** `IMPLEMENTED`
        - **Method:** L2 norm clipping
        - **Max norm:** `{cfg.dp_max_grad_norm}`
        - **Effect:** Bounds the sensitivity of each client's update before aggregation.
        - Each gradient update vector is scaled down if its L2 norm exceeds the threshold.
        """)

    with st.expander("🔊 Differential Privacy — DEMONSTRATION"):
        st.markdown(f"""
        - **Status:** `DEMONSTRATION`
        - **Mechanism:** Gaussian noise addition
        - **Enabled:** `{cfg.dp_enabled}`
        - **ε (epsilon):** `{cfg.dp_epsilon}`
        - **δ (delta):** `{cfg.dp_delta}`
        - **Noise multiplier:** `{cfg.dp_noise_multiplier}`
        > ⚠️ This is a demonstration of the DP mechanism. Production-grade differential privacy
        > requires formal mathematical verification and calibration to the specific threat model.
        """)

    with st.expander("🔒 Secure Aggregation — SIMULATION"):
        st.markdown(f"""
        - **Status:** `SIMULATION`
        - **Enabled:** `{cfg.secure_agg_enabled}`
        - **Simulation:** Additive masking (masks cancel after summation)
        > ⚠️ Production secure aggregation requires cryptographic protocols
        > (secret sharing, homomorphic encryption, or trusted execution environments).
        > This is a simulation for demonstration purposes only.
        """)

    st.markdown("### 📖 Privacy Definitions")
    st.markdown(
        """
        | Term | Classification | Description |
        |------|---------------|-------------|
        | Raw data locality | **IMPLEMENTED** | Data never leaves user device |
        | Update clipping | **IMPLEMENTED** | L2 norm clipping before aggregation |
        | Gaussian DP noise | **DEMONSTRATION** | Noise added to updates |
        | Secure aggregation | **SIMULATION** | Additive masking simulation |
        | Formal DP proof | **NOT IMPLEMENTED** | Requires rigorous mathematical analysis |
        | Cryptographic SecAgg | **NOT IMPLEMENTED** | Requires cryptographic protocols |
        """
    )

    st.info(
        "💡 **Core privacy guarantee:** Personal data stays with the client device. "
        "The federated server only sees aggregated, clipped, and optionally noisy adapter weight updates — "
        "never raw text, user profiles, or training samples."
    )
