import streamlit as st
from core.hardware import detect_hardware, EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD
from benchmark.metrics import collect_system_snapshot, BenchmarkError
import psutil


def _mode_badge(mode: str) -> str:
    styles = {
        EXECUTION_MODE_DEMO: ("background:#FF6B35;color:white", "DEMO"),
        EXECUTION_MODE_LIGHTWEIGHT: ("background:#1565C0;color:white", "LIGHTWEIGHT LOCAL"),
        EXECUTION_MODE_CLOUD: ("background:#2E7D32;color:white", "CLOUD TRAINING"),
    }
    style, label = styles.get(mode, ("background:#888;color:white", mode))
    return (
        f'<span style="{style};padding:4px 14px;border-radius:20px;'
        f'font-weight:700;font-size:0.82em;letter-spacing:0.04em">{label}</span>'
    )


def _metric_card(label: str, value: str, unit: str = "") -> None:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E8E8E8;border-left:4px solid #FF6B35;
                    padding:14px 18px;border-radius:8px;margin-bottom:8px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.06)">
            <div style="color:#888888;font-size:0.75em;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">{label}</div>
            <div style="color:#111111;font-size:1.25em;font-weight:700">
                {value}<span style="color:#AAAAAA;font-size:0.65em;margin-left:5px">{unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render() -> None:
    st.markdown(
        """
        <div style="padding:8px 0 20px 0">
            <h1 style="color:#FF6B35;font-size:2.2em;font-weight:800;margin-bottom:4px;letter-spacing:-0.01em">
                LokLLM
            </h1>
            <p style="color:#555555;font-size:1.05em;margin:0">
                Privacy-Preserving On-Device Personalization of LLMs via Federated QLoRA
            </p>
        </div>
        <hr style="border:none;border-top:1px solid #EBEBEB;margin-bottom:20px">
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
        st.markdown(
            f'<div style="padding-top:6px">{_mode_badge(mode)}</div>',
            unsafe_allow_html=True,
        )
    with col_msg:
        if mode == EXECUTION_MODE_DEMO:
            st.info("Running in **DEMO mode** — model not loaded. All outputs are simulated and clearly labelled.")
        elif mode == EXECUTION_MODE_LIGHTWEIGHT:
            st.success("Running in **LIGHTWEIGHT LOCAL mode** — small local model is available.")
        else:
            st.success("Running in **CLOUD TRAINING mode** — GPU detected.")

    _section_header("🖥️ System Hardware")

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

    _section_header("📊 Live Resource Usage")

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

    _section_header("🏗️ Architecture")

    col_arch_left, col_arch_right = st.columns([2, 1])
    with col_arch_left:
        st.markdown(
            """
            ```
            User Device (4 GB RAM, No GPU)
            ┌─────────────────────────────────────┐
            │  Synthetic Local Data               │
            │  Personalization Config             │
            │  Privacy Audit · Dashboard          │
            │  Hardware Monitoring                │
            └──────────────┬──────────────────────┘
                           │ Adapter Weights Only (~192 KB)
                           │ ❌ No Raw Data Transmitted
            ┌──────────────▼──────────────────────┐
            │  Cloud GPU Environment              │
            │  QLoRA Training (4-bit NF4)         │
            │  3-Client Flower FedAvg Simulation  │
            │  Adapter Export                     │
            └─────────────────────────────────────┘
            ```
            """
        )
    with col_arch_right:
        st.markdown(
            """
            <div style="background:#FFF8F5;border:1px solid #FFD5C2;border-radius:10px;
                        padding:16px 18px;margin-top:8px">
                <div style="color:#FF6B35;font-weight:700;font-size:0.9em;margin-bottom:10px">
                    💡 Core Principle
                </div>
                <div style="color:#333333;font-size:0.88em;line-height:1.65">
                    Personal data stays on the client.<br><br>
                    Only compressed LoRA adapter weights
                    (~192 KB) cross the network — never
                    raw conversations, medical records,
                    or financial data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _section_header("🗺️ Quick Navigation")

    nav_col1, nav_col2, nav_col3 = st.columns(3)
    nav_items = [
        ("🎯", "Personalization", "Synthetic data · Base vs. personalized responses"),
        ("🌐", "Federation", "FedAvg simulation · 3-client FL rounds"),
        ("🔒", "Privacy Audit", "Data boundary · DP · Secure aggregation"),
        ("📈", "Benchmarks", "RAM · CPU · Adapter size · Latency"),
        ("🏠", "Home", "This page — hardware profile and system overview"),
    ]
    cols = [nav_col1, nav_col2, nav_col3]
    for i, (emoji, name, desc) in enumerate(nav_items):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#FFFFFF;border:1px solid #E8E8E8;border-radius:8px;
                            padding:14px 16px;margin-bottom:10px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.05)">
                    <div style="font-size:1.4em;margin-bottom:4px">{emoji}</div>
                    <div style="font-weight:700;color:#111111;font-size:0.95em">{name}</div>
                    <div style="color:#888888;font-size:0.8em;margin-top:3px">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
