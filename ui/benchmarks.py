import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from benchmark.metrics import build_benchmark_result, collect_system_snapshot, NOT_AVAILABLE, NOT_MEASURED, BenchmarkError
from model.loader import get_model_info
from model.lora import get_lora_info
from federated.strategy import get_strategy_info
from config import get_config
import pandas as pd
import time


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
    color = "#888" if display in (NOT_AVAILABLE, NOT_MEASURED) else "#fff"
    st.markdown(
        f"""
        <div style="background:#1e1e1e;border-left:4px solid #FF6B35;padding:12px 16px;border-radius:6px">
            <div style="color:#aaa;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em">{label}</div>
            <div style="color:{color};font-size:1.1em;font-weight:600">{display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_ram_gauge(used_pct: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=used_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "RAM Used %", "font": {"color": "#aaa", "size": 14}},
        number={"suffix": "%", "font": {"color": "#fff", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": "#FF6B35"},
            "bgcolor": "#111",
            "steps": [
                {"range": [0, 60], "color": "#1a2a1a"},
                {"range": [60, 80], "color": "#2a2a1a"},
                {"range": [80, 100], "color": "#2a1a1a"},
            ],
            "threshold": {"line": {"color": "#f44336", "width": 3}, "thickness": 0.75, "value": 85},
        },
    ))
    fig.update_layout(paper_bgcolor="#111", font_color="#ccc", height=220, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        <h2 style="color:#FF6B35">📈 Benchmarks</h2>
        <p style="color:#aaa">
        Resource usage, model metrics, and communication payload measurements.
        Unavailable metrics are clearly marked rather than fabricated.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()

    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    try:
        snap = collect_system_snapshot()
        live_ok = True
    except BenchmarkError as e:
        st.warning(f"Live metrics unavailable: {e}")
        live_ok = False
        snap = None

    st.markdown("### 🖥️ Live System Metrics")
    if live_ok and snap:
        col1, col2 = st.columns([1, 1])
        with col1:
            _render_ram_gauge(snap.ram_used_percent)
        with col2:
            st.metric("RAM Total", f"{snap.ram_total_gb:.2f} GB")
            st.metric("RAM Available", f"{snap.ram_available_gb:.2f} GB")
            st.metric("CPU Usage", f"{snap.cpu_percent:.1f}%")
            st.progress(min(snap.cpu_percent / 100, 1.0))
    else:
        st.info("Live metrics could not be retrieved.")

    st.markdown("### 🤖 Model Metrics")
    model_info = get_model_info()
    lora_info = get_lora_info(cfg)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        _metric_card("Model Loaded", "Yes" if model_info["loaded"] else "No (DEMO mode)")
    with col_m2:
        _metric_card("Total Parameters", model_info.get("total_parameters") or NOT_MEASURED)
    with col_m3:
        _metric_card("Trainable Params", model_info.get("trainable_parameters") or NOT_MEASURED)
    with col_m4:
        _metric_card("LoRA Rank (r)", lora_info["r"])

    st.markdown("### 📦 Adapter & Communication")
    result = build_benchmark_result(
        mode="DEMO",
        federated_rounds=st.session_state.get("fed_rounds_done", 0),
    )

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        _metric_card("Adapter Size (KB)", result.adapter_size_kb)
    with col_a2:
        _metric_card("Communication Payload (KB)", result.communication_payload_kb)
    with col_a3:
        _metric_card("Federated Rounds", result.federated_rounds)
    with col_a4:
        _metric_card("Trainable Params (Demo)", result.trainable_parameters)

    st.markdown("### ⏱️ Performance")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        _metric_card("Inference Latency", result.inference_latency_ms)
    with col_p2:
        _metric_card("Training Time", result.training_time_seconds)

    st.markdown("### 📊 Adapter Size vs. Full Model (Demo)")
    fig_bar = go.Figure(go.Bar(
        x=["Full Model\n(~85M params)", "LoRA Adapter\n(~49K params)"],
        y=[85_000_000, 49_152],
        marker_color=["#2196F3", "#FF6B35"],
        text=["85,000,000", "49,152"],
        textposition="outside",
    ))
    fig_bar.update_layout(
        paper_bgcolor="#111",
        plot_bgcolor="#1a1a1a",
        font_color="#ccc",
        yaxis=dict(type="log", title="Parameters (log scale)", color="#888"),
        xaxis=dict(color="#888"),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        title=dict(text="[DEMO] Parameter Efficiency of LoRA", font=dict(color="#FF6B35")),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📡 Communication Efficiency")
    labels = ["Raw Model Transfer\n(hypothetical)", "LoRA Adapter\n(actual)"]
    sizes = [340_000, 192]
    fig_comm = px.pie(
        values=sizes,
        names=labels,
        color_discrete_sequence=["#2196F3", "#FF6B35"],
        title="[DEMO] Data Transmitted: Full Model vs LoRA Adapter",
    )
    fig_comm.update_layout(paper_bgcolor="#111", font_color="#ccc", height=300)
    st.plotly_chart(fig_comm, use_container_width=True)

    st.markdown("### 📋 Full Benchmark Table")
    rows = [
        ("RAM Total (GB)", result.ram_total_gb),
        ("RAM Used (GB)", result.ram_used_gb),
        ("RAM Used (%)", f"{result.ram_used_percent}%"),
        ("CPU Usage (%)", f"{result.cpu_percent}%"),
        ("Inference Latency (ms)", result.inference_latency_ms),
        ("Adapter Size (KB)", result.adapter_size_kb),
        ("Trainable Parameters", result.trainable_parameters),
        ("Total Parameters", result.total_parameters),
        ("Communication Payload (KB)", result.communication_payload_kb),
        ("Federated Rounds", result.federated_rounds),
        ("Training Time", result.training_time_seconds),
        ("Mode", result.mode),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df["Value"] = df["Value"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if result.notes:
        for note in result.notes:
            st.caption(f"ℹ️ {note}")
