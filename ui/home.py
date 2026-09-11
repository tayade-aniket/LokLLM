import streamlit as st
from core.hardware import detect_hardware, EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD
from benchmark.metrics import collect_system_snapshot, BenchmarkError
import psutil


def _mode_badge(mode: str) -> str:
    colors = {
        EXECUTION_MODE_DEMO: "#FF6B35",
        EXECUTION_MODE_LIGHTWEIGHT: "#2196F3",
        EXECUTION_MODE_CLOUD: "#4CAF50",
    }
    color = colors.get(mode, "#888888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:4px;font-weight:bold;font-size:0.85em">{mode}</span>'


def _metric_card(label: str, value: str, unit: str = "") -> None:
    st.markdown(
        f"""
        <div style="background:#1e1e1e;border-left:4px solid #FF6B35;padding:12px 16px;border-radius:6px;margin-bottom:8px">
            <div style="color:#aaa;font-size:0.8em;text-transform:uppercase;letter-spacing:0.05em">{label}</div>
            <div style="color:#fff;font-size:1.3em;font-weight:600">{value}<span style="color:#888;font-size:0.7em;margin-left:4px">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        """
        <h1 style="color:#FF6B35;margin-bottom:0">PRIVFEDQLORA</h1>
        <p style="color:#aaa;margin-top:4px;font-size:1.1em">
        Privacy-Preserving On-Device Personalization of LLMs via Federated QLoRA
        </p>
        <hr style="border-color:#333;margin:16px 0">
        """,
        unsafe_allow_html=True,
    )

    try:
        profile = detect_hardware()
        mode = profile.execution_mode
    except Exception as e:
        st.error(f"Hardware detection failed: {e}")
        return

    col_mode, col_msg = st.columns([1, 3])
    with col_mode:
        st.markdown(_mode_badge(mode), unsafe_allow_html=True)
    with col_msg:
        if mode == EXECUTION_MODE_DEMO:
            st.info("Running in **DEMO mode** — model not loaded. All outputs are simulated and clearly labelled.")
        elif mode == EXECUTION_MODE_LIGHTWEIGHT:
            st.success("Running in **LIGHTWEIGHT LOCAL mode** — small local model available.")
        else:
            st.success("Running in **CLOUD TRAINING mode** — GPU available.")

    st.markdown("### 🖥️ System Hardware")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("CPU", profile.cpu_name[:28] if len(profile.cpu_name) > 28 else profile.cpu_name)
    with col2:
        _metric_card("Total RAM", f"{profile.ram_total_gb}", "GB")
    with col3:
        _metric_card("Available RAM", f"{profile.ram_available_gb}", "GB")
    with col4:
        _metric_card("Free Disk", f"{profile.disk_free_gb}", "GB")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        _metric_card("CPU Cores (Logical)", str(profile.cpu_cores_logical))
    with col6:
        _metric_card("OS", profile.os_name)
    with col7:
        _metric_card("GPU", "Yes" if profile.gpu_available else "No")
    with col8:
        _metric_card("CUDA", profile.cuda_version if profile.cuda_available else "Not Available")

    st.markdown("### 📊 Live Resource Usage")
    try:
        snap = collect_system_snapshot()
        col_ram, col_cpu = st.columns(2)
        with col_ram:
            st.metric("RAM Used", f"{snap.ram_used_percent:.1f}%")
            st.progress(snap.ram_used_percent / 100)
        with col_cpu:
            st.metric("CPU Usage", f"{snap.cpu_percent:.1f}%")
            st.progress(min(snap.cpu_percent / 100, 1.0))
    except BenchmarkError as e:
        st.warning(f"Could not read live metrics: {e}")

    st.markdown("### 🏗️ Architecture")
    st.markdown(
        """
        ```
        User Device (4 GB RAM, No GPU)
        ┌─────────────────────────────────────┐
        │  Synthetic Local Data               │
        │  Personalization Config             │
        │  Privacy Audit                      │
        │  Streamlit Dashboard                │
        │  Hardware Monitoring                │
        └──────────────┬──────────────────────┘
                       │ Adapter Weights Only
                       │ (No Raw Data Transmitted)
        ┌──────────────▼──────────────────────┐
        │  Cloud GPU Environment              │
        │  QLoRA Training (4-bit NF4)         │
        │  3-Client Flower Simulation         │
        │  FedAvg Aggregation                 │
        │  Adapter Export                     │
        └─────────────────────────────────────┘
        ```
        """
    )

    st.markdown("### 💡 Core Message")
    st.info(
        "**Personal data stays with the client.** Model knowledge is collaboratively improved "
        "through parameter-efficient LoRA updates rather than sharing raw data. "
        "Only compressed adapter weights (~192 KB) cross the network boundary."
    )

    st.markdown("### 🗺️ Navigation")
    st.markdown(
        """
        | Page | Description |
        |------|-------------|
        | 🏠 Home | Hardware info and system overview |
        | 🎯 Personalization | Synthetic data and personalized inference |
        | 🌐 Federation | Federated learning round control |
        | 🔒 Privacy Audit | Privacy boundary visualization |
        | 📈 Benchmarks | Resource and performance metrics |
        """
    )
