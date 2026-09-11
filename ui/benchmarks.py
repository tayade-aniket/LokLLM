import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from benchmark.metrics import build_benchmark_result, collect_system_snapshot, NOT_AVAILABLE, NOT_MEASURED, BenchmarkError
from model.loader import get_model_info
from model.lora import get_lora_info
from federated.strategy import get_strategy_info
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


def _fmt(value) -> str:
    if value in (NOT_AVAILABLE, NOT_MEASURED):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _metric_card(label: str, value) -> None:
    display = _fmt(value)
    color = "#888888" if display in (NOT_AVAILABLE, NOT_MEASURED) else "#111111"
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E8E8E8;border-left:4px solid #FF6B35;
                    padding:14px 16px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:8px">
            <div style="color:#888888;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">{label}</div>
            <div style="color:{color};font-size:1.15em;font-weight:700">{display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_ram_gauge(used_pct: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=used_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "RAM Used", "font": {"color": "#333333", "size": 14, "family": "sans-serif"}},
        number={"suffix": "%", "font": {"color": "#111111", "size": 22, "family": "sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888888"},
            "bar": {"color": "#FF6B35"},
            "bgcolor": "#F5F5F5",
            "steps": [
                {"range": [0, 60], "color": "#E8F5E9"},
                {"range": [60, 80], "color": "#FFF3E0"},
                {"range": [80, 100], "color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": "#D32F2F", "width": 3}, "thickness": 0.75, "value": 85},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font_color="#111111",
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <div style="padding:8px 0 20px 0">
            <h2 style="color:#FF6B35;font-weight:800;margin-bottom:4px">📈 Benchmarks</h2>
            <p style="color:#555555;font-size:1.0em;margin:0">
                Resource consumption, parameter efficiency, and communication payload measurements.
                Unavailable metrics are clearly indicated rather than fabricated.
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()

    if st.button("🔄 Refresh Benchmarks"):
        st.rerun()

    try:
        snap = collect_system_snapshot()
        live_ok = True
    except BenchmarkError as e:
        st.warning(f"Live metrics unavailable: {e}")
        live_ok = False
        snap = None

    _section_header("🖥️ System Resource Utilization")
    if live_ok and snap:
        col1, col2 = st.columns([1, 1])
        with col1:
            _render_ram_gauge(snap.ram_used_percent)
        with col2:
            st.metric("RAM Total", f"{snap.ram_total_gb:.2f} GB")
            st.metric("RAM Available", f"{snap.ram_available_gb:.2f} GB")
            st.metric("CPU Load", f"{snap.cpu_percent:.1f}%")
            st.progress(min(snap.cpu_percent / 100, 1.0))
    else:
        st.info("Live system metrics are currently unavailable.")

    _section_header("🤖 Model & LoRA Parameters")
    model_info = get_model_info()
    lora_info = get_lora_info(cfg)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        _metric_card("Model Status", "Loaded" if model_info["loaded"] else "DEMO Mode")
    with col_m2:
        _metric_card("Total Parameters", model_info.get("total_parameters") or NOT_MEASURED)
    with col_m3:
        _metric_card("Trainable Params", model_info.get("trainable_parameters") or NOT_MEASURED)
    with col_m4:
        _metric_card("LoRA Rank (r)", lora_info["r"])

    _section_header("📦 Adapter & Transmission Efficiency")
    result = build_benchmark_result(
        mode="DEMO",
        federated_rounds=st.session_state.get("fed_rounds_done", 0),
    )

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        _metric_card("Adapter Size (KB)", result.adapter_size_kb)
    with col_a2:
        _metric_card("Comm. Payload (KB)", result.communication_payload_kb)
    with col_a3:
        _metric_card("Federated Rounds", result.federated_rounds)
    with col_a4:
        _metric_card("Trainable Params (Demo)", result.trainable_parameters)

    _section_header("⏱️ Latency & Execution Time")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        _metric_card("Inference Latency", result.inference_latency_ms)
    with col_p2:
        _metric_card("Training Duration", result.training_time_seconds)

    _section_header("📊 Parameter & Communication Comparison")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_bar = go.Figure(go.Bar(
            x=["Full Model\n(~85M params)", "LoRA Adapter\n(~49K params)"],
            y=[85_000_000, 49_152],
            marker_color=["#1565C0", "#FF6B35"],
            text=["85,000,000", "49,152"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FAFAFA",
            font_color="#111111",
            yaxis=dict(type="log", title="Parameters (log scale)", gridcolor="#EEEEEE"),
            xaxis=dict(color="#333333"),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            title=dict(text="Parameter Efficiency (LoRA vs Full Model)", font=dict(color="#FF6B35", size=13)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        labels = ["Full Model Weights", "LoRA Adapter Weights"]
        sizes = [340_000, 192]
        fig_comm = px.pie(
            values=sizes,
            names=labels,
            color_discrete_sequence=["#1565C0", "#FF6B35"],
            title="Communication Payload Comparison (KB)",
        )
        fig_comm.update_layout(
            paper_bgcolor="#FFFFFF",
            font_color="#111111",
            height=300,
            title_font_color="#FF6B35",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_comm, use_container_width=True)

    _section_header("📋 Comprehensive Metrics Table")
    rows = [
        ("RAM Total (GB)", result.ram_total_gb),
        ("RAM Used (GB)", result.ram_used_gb),
        ("RAM Used (%)", f"{result.ram_used_percent}%"),
        ("CPU Load (%)", f"{result.cpu_percent}%"),
        ("Inference Latency (ms)", result.inference_latency_ms),
        ("Adapter Size (KB)", result.adapter_size_kb),
        ("Trainable Parameters", result.trainable_parameters),
        ("Total Parameters", result.total_parameters),
        ("Communication Payload (KB)", result.communication_payload_kb),
        ("Federated Rounds", result.federated_rounds),
        ("Training Time", result.training_time_seconds),
        ("Execution Mode", result.mode),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df["Value"] = df["Value"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if result.notes:
        for note in result.notes:
            st.caption(f"ℹ️ {note}")
