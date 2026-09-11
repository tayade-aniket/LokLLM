import streamlit as st
import plotly.graph_objects as go
from privacy.audit import build_audit_report, get_privacy_boundary_summary
from privacy.clipping import get_clipping_info
from privacy.differential_privacy import get_dp_info
from privacy.secure_aggregation import get_secure_agg_info
from config import get_config


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


def _status_badge(status: str) -> str:
    styles = {
        "IMPLEMENTED": "background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7",
        "DEMONSTRATION": "background:#FFF3E0;color:#E65100;border:1px solid #FFCC80",
        "SIMULATION": "background:#E3F2FD;color:#1565C0;border:1px solid #90CAF9",
        "NOT IMPLEMENTED": "background:#FFEBEE;color:#C62828;border:1px solid #EF9A9A",
        "YES": "background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7",
        "NO": "background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7",
        "Enabled": "background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7",
        "Disabled": "background:#F5F5F5;color:#757575;border:1px solid #E0E0E0",
        "Simulation": "background:#E3F2FD;color:#1565C0;border:1px solid #90CAF9",
    }
    style = styles.get(status, "background:#F5F5F5;color:#424242;border:1px solid #E0E0E0")
    return f'<span style="{style};padding:3px 10px;border-radius:12px;font-size:0.75em;font-weight:700">{status}</span>'


def _privacy_metric_row(label: str, value: str, badge: str = None) -> None:
    badge_html = _status_badge(badge) if badge else ""
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:12px 18px;border-bottom:1px solid #EEEEEE;background:#FFFFFF">
            <span style="color:#333333;font-size:0.92em;font-weight:500">{label}</span>
            <span style="color:#111111;font-weight:600;display:flex;align-items:center;gap:10px">
                {value} {badge_html}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_flow_diagram() -> None:
    fig = go.Figure()

    nodes = [
        (0.1, 0.5, "🔐 Raw Data\n(Local)", "#C62828", "#FFEBEE"),
        (0.35, 0.5, "📱 Local Client\nDevice", "#FF6B35", "#FFF3E0"),
        (0.6, 0.5, "✂️ Clip +\nDP Noise", "#E65100", "#FFF8E1"),
        (0.85, 0.5, "☁️ Federated\nServer", "#1565C0", "#E3F2FD"),
    ]

    for x, y, label, color, bg in nodes:
        fig.add_shape(
            type="circle", x0=x - 0.08, y0=y - 0.2, x1=x + 0.08, y1=y + 0.2,
            line=dict(color=color, width=2), fillcolor=bg,
        )
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(color=color, size=10, family="sans-serif"))

    arrows = [
        (0.18, 0.5, 0.27, 0.5, "#2E7D32", "Local only\n(0 bytes uploaded)"),
        (0.43, 0.5, 0.52, 0.5, "#FF6B35", "Adapter Δw only\n(~192 KB)"),
        (0.68, 0.5, 0.77, 0.5, "#1565C0", "Noisy / Clipped Δw"),
    ]
    for x0, y0, x1, y1, color, label in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y",
            arrowhead=2, arrowsize=1.5, arrowcolor=color,
            text=label, font=dict(color=color, size=9),
            xanchor="center", yanchor="bottom", yshift=24,
        )

    fig.add_shape(
        type="rect", x0=0.01, y0=0.12, x1=0.20, y1=0.88,
        line=dict(color="#C62828", dash="dot", width=1.5), fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=0.105, y=0.07, text="🚫 Never leaves device",
        showarrow=False, font=dict(color="#C62828", size=10),
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=290,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <div style="padding:8px 0 20px 0">
            <h2 style="color:#FF6B35;font-weight:800;margin-bottom:4px">🔒 Privacy Audit</h2>
            <p style="color:#555555;font-size:1.0em;margin:0">
                Transparency report on what data is transmitted, what stays local, and which privacy mechanisms are active.
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    report = build_audit_report(mode="DEMO")
    boundary = get_privacy_boundary_summary()
    clipping_info = get_clipping_info(cfg.dp_max_grad_norm)
    dp_info = get_dp_info(cfg)
    sec_agg_info = get_secure_agg_info(cfg.secure_agg_enabled)

    _section_header("🔍 Privacy Boundary — Data Flow")
    _render_data_flow_diagram()

    _section_header("📋 Audit Report")
    st.markdown('<div style="background:#FFFFFF;border:1px solid #E8E8E8;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05)">', unsafe_allow_html=True)
    _privacy_metric_row("Raw personal data uploaded", "0 bytes", "NO")
    _privacy_metric_row("Raw dataset transmitted", "NO", "NO")
    _privacy_metric_row("Local training performed", "YES (edge or simulated)", "YES")
    _privacy_metric_row("Adapter weights transmitted", "YES — LoRA adapter only (~192 KB)", "YES")
    _privacy_metric_row("Update clipping", f"L2 norm ≤ {cfg.dp_max_grad_norm}", clipping_info["status"])
    _privacy_metric_row("Differential privacy", f"ε={cfg.dp_epsilon}, δ={cfg.dp_delta}", dp_info["status"])
    _privacy_metric_row("Secure aggregation", sec_agg_info["description"][:45] + "…", sec_agg_info["status"])
    st.markdown("</div>", unsafe_allow_html=True)

    _section_header("🛡️ Mechanism Details")

    with st.expander("✂️ Update Clipping — IMPLEMENTED"):
        st.markdown(f"""
        - **Status:** `IMPLEMENTED`
        - **Method:** L2 norm clipping
        - **Max norm:** `{cfg.dp_max_grad_norm}`
        - **Effect:** Limits the sensitivity of each client's update vector prior to aggregation.
        - Updates with L2 norm exceeding threshold are scaled down proportionally.
        """)

    with st.expander("🔊 Differential Privacy — DEMONSTRATION"):
        st.markdown(f"""
        - **Status:** `DEMONSTRATION`
        - **Mechanism:** Gaussian noise addition
        - **Enabled:** `{cfg.dp_enabled}`
        - **ε (epsilon):** `{cfg.dp_epsilon}`
        - **δ (delta):** `{cfg.dp_delta}`
        - **Noise multiplier:** `{cfg.dp_noise_multiplier}`
        > ℹ️ *Note:* This demonstrates the Gaussian DP noise mechanism on adapter vectors. Production guarantees require formal mathematical accounting and calibrated noise levels.
        """)

    with st.expander("🔒 Secure Aggregation — SIMULATION"):
        st.markdown(f"""
        - **Status:** `SIMULATION`
        - **Enabled:** `{cfg.secure_agg_enabled}`
        - **Simulation:** Additive zero-sum masking
        > ℹ️ *Note:* Simulates mask cancellation over federated sums. Production deployment requires cryptographic multi-party computation or secure enclave verification.
        """)

    _section_header("📖 Privacy Classification")
    st.markdown(
        """
        | Mechanism | Classification | Verification Status |
        |---|---|---|
        | Raw Data Locality | **IMPLEMENTED** | Zero raw bytes transmitted across network |
        | Update Norm Clipping | **IMPLEMENTED** | Strict L2 norm threshold applied |
        | Gaussian DP Noise | **DEMONSTRATION** | Additive noise calibrated to multiplier |
        | Secure Aggregation | **SIMULATION** | Additive masking demonstration |
        | Formal DP Proof | **NOT IMPLEMENTED** | Requires end-to-end differential privacy audit |
        | Cryptographic MPC | **NOT IMPLEMENTED** | Requires cryptographic protocol integration |
        """
    )

    st.info(
        "💡 **Core Privacy Guarantee:** Personal data stays with the client device. "
        "The federated coordinator only observes aggregated, bounded, and optionally perturbed adapter weight vectors."
    )
