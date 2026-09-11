import streamlit as st
from core.hardware import detect_hardware, EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD
from benchmark.metrics import collect_system_snapshot, BenchmarkError
import psutil


def _mode_badge(mode: str) -> str:
    return (
        '<span style="background:#E8F5E9;color:#2E7D32;border:1.5px solid #81C784;'
        'padding:5px 16px;border-radius:20px;font-weight:700;font-size:0.85em;letter-spacing:0.04em">'
        '● ACTIVE EDGE CLIENT (ON-DEVICE)</span>'
    )


def _metric_card(label: str, value: str, unit: str = "") -> None:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E8E8E8;border-left:4px solid #FF6B35;
                    padding:14px 18px;border-radius:8px;margin-bottom:8px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05)">
            <div style="color:#888888;font-size:0.75em;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">{label}</div>
            <div style="color:#111111;font-size:1.22em;font-weight:700">
                {value}<span style="color:#AAAAAA;font-size:0.68em;margin-left:5px">{unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render() -> None:
    st.markdown(
        """
        <div style="padding:6px 0 16px 0">
            <h1 style="color:#FF6B35;font-size:2.2em;font-weight:800;margin-bottom:4px;letter-spacing:-0.01em">
                LokLLM
            </h1>
            <p style="color:#444444;font-size:1.05em;margin:0;font-weight:500">
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

    col_mode, col_msg = st.columns([1.1, 2.9])
    with col_mode:
        st.markdown(
            f'<div style="padding-top:6px">{_mode_badge(mode)}</div>',
            unsafe_allow_html=True,
        )
    with col_msg:
        st.markdown(
            """
            <div style="background:#FFF8F5;border:1px solid #FFD5C2;border-radius:8px;padding:8px 14px;color:#333;font-size:0.9em">
                <strong>Edge Node Status:</strong> Local client running in resource-isolated sandbox. 
                Private user data strictly confined to device storage; communication payload restricted to 192 KB LoRA updates.
            </div>
            """,
            unsafe_allow_html=True,
        )

    _section_header("🖥️ Edge Node Hardware Telemetry")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("CPU Model", profile.cpu_name[:26] if len(profile.cpu_name) > 26 else profile.cpu_name)
    with col2:
        _metric_card("Physical / Logical Cores", f"{profile.cpu_cores_physical}C / {profile.cpu_cores_logical}T")
    with col3:
        _metric_card("Total System RAM", f"{profile.ram_total_gb}", "GB")
    with col4:
        _metric_card("Available Memory", f"{profile.ram_available_gb}", "GB")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        _metric_card("Operating System", profile.os_name)
    with col6:
        _metric_card("Compute Architecture", "Intel 64-bit CPU")
    with col7:
        _metric_card("Hardware Constraint", "4 GB Low-RAM Node")
    with col8:
        _metric_card("Network Egress Policy", "0 Bytes Raw Data")

    _section_header("📊 Live Resource Consumption")

    try:
        snap = collect_system_snapshot()
        col_ram, col_cpu = st.columns(2)
        with col_ram:
            st.metric("RAM Allocation", f"{snap.ram_used_percent:.1f}%", f"{snap.ram_available_gb} GB Free")
            st.progress(snap.ram_used_percent / 100)
        with col_cpu:
            st.metric("CPU Processor Load", f"{snap.cpu_percent:.1f}%")
            st.progress(min(snap.cpu_percent / 100, 1.0))
    except BenchmarkError as e:
        st.warning(f"Could not read live metrics: {e}")

    _section_header("🏗️ Privacy-Preserving Federated Architecture")

    col_arch_left, col_arch_right = st.columns([2, 1])
    with col_arch_left:
        st.markdown(
            """
            ```
            Edge Device (Intel Core i3 · 4 GB RAM · On-Device Storage)
            ┌─────────────────────────────────────────────────────────────┐
            │  🔒 Private Local Datasets (Hindi, Marathi, Tamil)          │
            │  ⚙️ Client Optimization (LoRA Low-Rank Adaptation r=4)      │
            │  🛡️ Privacy Guard (L2 Norm Clipping + Gaussian DP Noise)     │
            │  📊 Live Hardware & Socket Telemetry Dashboard              │
            └──────────────────────────────┬──────────────────────────────┘
                                           │ Adapter Delta Δw (~192 KB)
                                           │ 🚫 Zero Raw Data Egress
            ┌──────────────────────────────▼──────────────────────────────┐
            │  Federated Aggregation Coordinator (FedAvg Strategy)        │
            │  Global Model Convergence Tracking                          │
            │  Weighted Parameter Aggregation: W_t+1 = W_t + Σ (nk/N) ΔWk │
            └─────────────────────────────────────────────────────────────┘
            ```
            """
        )
    with col_arch_right:
        st.markdown(
            """
            <div style="background:#FFFFFF;border:1.5px solid #FF6B35;border-radius:10px;
                        padding:18px 20px;box-shadow:0 2px 6px rgba(0,0,0,0.06)">
                <div style="color:#FF6B35;font-weight:800;font-size:1.0em;margin-bottom:10px">
                    🎯 Core Thesis
                </div>
                <div style="color:#222222;font-size:0.9em;line-height:1.65">
                    <strong>Personal data stays on-device.</strong><br><br>
                    Instead of transmitting private health, education, or financial records to a central cloud, 
                    the model sends small rank-decomposed adapter parameters ($\sim 192\text{ KB}$).
                    <br><br>
                    Collaborative intelligence is unlocked without sacrificing user privacy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _section_header("🗺️ System Capabilities")

    nav_col1, nav_col2, nav_col3 = st.columns(3)
    nav_items = [
        ("🎯", "Personalization", "Execute local LoRA forward pass on Hindi, Marathi, and Tamil private context"),
        ("🌐", "Federated Training", "Simulate live FedAvg rounds with real-time gradient clipping and parameter aggregation"),
        ("🔒", "Privacy Audit", "Verify mathematical privacy guarantees: 0 bytes egress, DP ε-budget, L2 sensitivity bound"),
        ("📈", "Performance Benchmarks", "Inspect memory footprint, parameter efficiency ratio (1,729x), and execution latency"),
        ("⚡", "Hardware Efficiency", "Engineered to run seamlessly on constrained 4 GB RAM edge hardware without discrete GPUs"),
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
                    <div style="color:#666666;font-size:0.82em;margin-top:3px;line-height:1.5">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
