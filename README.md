<div align="center">

# 🔒 PRIVFEDQLORA

### Privacy-Preserving On-Device Personalization of LLMs via Federated QLoRA

*A hackathon research prototype — built to survive on 4 GB RAM, no GPU, no excuses.*

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-FedAvg-pink?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

<br>

![Tests](https://img.shields.io/badge/Tests-60%20Passed-4CAF50?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%2010-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![RAM](https://img.shields.io/badge/RAM-4%20GB%20(No%20GPU)-FF6B35?style=for-the-badge)
![Privacy](https://img.shields.io/badge/Raw%20Data%20Uploaded-0%20bytes-success?style=for-the-badge)

</div>

---

## What is this?

Imagine you're a doctor in a rural Hindi-speaking district. Your phone's AI assistant could be incredibly helpful — but only if it understands your specific medical context and language. The catch? Training it on your patients' data means their information leaves your hands. That's not acceptable.

PRIVFEDQLORA solves this. It lets a language model learn from your private local data — without ever sending that data anywhere. Only a tiny set of model weight updates (about 192 KB, smaller than most profile photos) gets shared. The raw data stays exactly where it belongs: on your device.

We built this for three real-world Indian use cases:

| Client | Domain | Language |
|--------|--------|----------|
| 🏥 Client 1 | Rural Healthcare | Hindi |
| 📚 Client 2 | School Education | Marathi |
| 💰 Client 3 | Financial Literacy | Tamil |

---

## The Core Idea

```
Your private data trains a tiny adapter on your device.
The adapter (not the data) gets shared.
Three such adapters get merged via FedAvg.
Everyone's model improves. Nobody's data moves.
```

That's it. Federated Learning + LoRA + QLoRA, working together.

---

## Why these three technologies?

### On-Device Training — because your data is yours

Raw medical queries, student notes, and financial records belong to the person who created them. The moment they leave a device, you've lost control. On-device training means the model comes to the data — not the other way around.

### LoRA — because full fine-tuning is overkill

Fine-tuning a billion-parameter model from scratch updates every single weight. That's slow, memory-hungry, and wasteful when you only need domain adaptation. LoRA adds two small matrices to each attention layer — updating less than 0.06% of total parameters while capturing the same knowledge shift.

|  | Full Fine-Tuning | LoRA (r=4) |
|--|--|--|
| Parameters updated | ~85 million | ~49,000 |
| Network payload | ~340 MB | ~192 KB |
| GPU RAM needed | 32+ GB | 6–8 GB |

### QLoRA — because 6 GB is still too much for most people

QLoRA compresses the frozen base model to 4-bit NF4 precision before training. That halves the memory again, making it feasible to fine-tune a 7B model on a single consumer GPU. It's clever — you keep the quality, you shed the weight.

### Federated Learning — because collaboration shouldn't require trust

A healthcare clinic, a school, and a bank can all improve the same model — even though they'd never share their databases with each other. Federated Learning makes that possible. Each party trains locally, sends a small update, and gets back a better global model. The server never sees a single patient record or student grade.

---

## How the privacy actually works

```
┌─────────────────────────────┐
│   Your Device               │
│                             │
│  Private Data               │
│       ↓                     │
│  Local Training             │
│       ↓                     │
│  L2 Norm Clipping           │  ← caps how much influence
│       ↓                     │    one person's data has
│  + Gaussian Noise (DP)      │  ← adds randomness to obscure
│       ↓                     │    individual contributions
│  Adapter Update Δw          │
└──────────────┬──────────────┘
               │  ~192 KB
               │  (not raw data)
               ▼
      Federated Server
      FedAvg(Δw₁, Δw₂, Δw₃)
               │
               ▼
      Improved global adapter
```

| Mechanism | Status | What it means |
|-----------|--------|---------------|
| Raw data never leaves device | ✅ IMPLEMENTED | Zero bytes transmitted |
| L2 norm update clipping | ✅ IMPLEMENTED | Bounds each client's influence |
| Gaussian DP noise | 🔶 DEMONSTRATION | Adds ε-δ differential privacy noise |
| Secure aggregation | 🔵 SIMULATION | Additive masking — crypto version not included |
| Formal DP proof | ❌ NOT INCLUDED | Would need rigorous mathematical auditing |

We don't pretend this is production-grade cryptographic privacy. What we do show is the correct architecture and the right mechanisms — applied honestly and labelled clearly.

---

## System Architecture

```
Local Machine (Intel Core i3 · 4 GB RAM · No GPU)
┌──────────────────────────────────────────────────────────┐
│  Streamlit Dashboard     Hardware Monitoring             │
│  Synthetic Data Gen      Privacy Audit                   │
│  DEMO Inference          Benchmark Visualization         │
│  Federation Control UI   Experiment Configuration        │
└─────────────────────┬────────────────────────────────────┘
                      │  LoRA adapter weights only (~192 KB)
                      │  ❌ No raw data. Ever.
                      ▼
Cloud GPU Environment
┌──────────────────────────────────────────────────────────┐
│  4-bit NF4 QLoRA Training    Flower FedAvg Simulation    │
│  3-Client FL Orchestration   Adapter Merge & Export      │
└──────────────────────────────────────────────────────────┘
```

### Execution Modes

The app detects your hardware at startup and picks the right mode automatically. Nothing breaks.

| Mode | Triggered when | What happens |
|------|---------------|--------------|
| `DEMO` | RAM < 2.5 GB or no model | Simulated responses, clearly labelled |
| `LIGHTWEIGHT_LOCAL` | RAM ≥ 2.5 GB, no GPU | Tiny local model runs for real |
| `CLOUD_TRAINING` | CUDA available | Full QLoRA pipeline active |

---

## Project Layout

```
PRIVFEDQLORA/
├── app.py                     Streamlit entry point
├── config.py                  All settings, env-driven
├── requirements.txt
├── .env.example
│
├── core/
│   ├── hardware.py            Detects RAM, CPU, GPU, sets execution mode
│   ├── logger.py              Structured logging (no secrets, no raw data)
│   └── exceptions.py         Custom exception hierarchy
│
├── data/
│   ├── synthetic_data.py      30 samples across 3 language-domain clients
│   ├── dataset.py             LocalDataset wrapper with lazy iteration
│   └── preprocessing.py      Tokenization helpers, small batch safe
│
├── model/
│   ├── loader.py              Lazy loading, RAM-gated, graceful fallback
│   ├── lora.py                LoRA config + 4-bit QLoRA BnB config
│   ├── inference.py           Real inference + DEMO stub with domain detection
│   └── adapter.py             Save, load, inspect adapter files
│
├── federated/
│   ├── client.py              Flower NumPyClient + pure-Python DemoClient
│   ├── server.py              Flower server config builder
│   ├── strategy.py            FedAvg wrapper
│   └── simulation.py         Orchestrates demo or real FL rounds
│
├── privacy/
│   ├── clipping.py            L2 norm clipping — actually implemented
│   ├── differential_privacy.py  Gaussian noise — demonstration
│   ├── secure_aggregation.py    Additive masking — simulation
│   └── audit.py               Builds the privacy audit report
│
├── benchmark/
│   └── metrics.py             RAM, CPU, latency, adapter size — no fabrication
│
├── ui/
│   ├── home.py                Hardware info, live metrics, system overview
│   ├── personalization.py     Synthetic data viewer + inference demo
│   ├── federation.py          FL round control + charts
│   ├── privacy.py             Privacy audit dashboard
│   └── benchmarks.py         Gauge charts, comparison plots, metrics table
│
├── cloud/                     GPU-only scripts, independent of Streamlit
│   ├── train_qlora.py         Single-client QLoRA training
│   ├── run_federated.py       3-client Flower simulation
│   └── export_adapter.py     Merge and export adapter with metadata
│
└── tests/                     60 tests, all pass on 4 GB / no GPU
    ├── test_hardware.py
    ├── test_data.py
    ├── test_privacy.py
    └── test_smoke.py
```

---

## Getting Started

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd PRIVFEDQLORA
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

**Minimal install — runs the full dashboard in DEMO mode (no GPU, no large downloads):**

```bash
pip install streamlit plotly psutil numpy pandas python-dotenv
```

**Full install — enables local model loading and cloud scripts:**

```bash
pip install -r requirements.txt
```

### 4. Configure your environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
```

Key settings in `.env`:

```env
MODEL_MODE=lightweight          # or: cloud
LOCAL_MODEL_ID=sshleifer/tiny-gpt2
CLOUD_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DP_ENABLED=true
HF_TOKEN=                       # optional, for gated models
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` — the app detects your hardware automatically and picks the right mode.

---

## Cloud Training (GPU machine only)

These scripts are designed to run on a remote GPU environment. They are completely independent of the Streamlit UI.

**Train a single client:**

```bash
python cloud/train_qlora.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --client-id client_1 \
  --output-dir storage/adapters \
  --lora-r 4 --lora-alpha 16
```

**Run the full 3-client federated simulation:**

```bash
python cloud/run_federated.py \
  --num-rounds 1 \
  --output-dir storage/results
```

**Export the trained adapter:**

```bash
python cloud/export_adapter.py \
  --adapter-dir storage/adapters/client_1 \
  --output-dir storage/adapters/exported

# No GPU? Create a demo record for the dashboard:
python cloud/export_adapter.py --demo --output-dir storage/adapters
```

---

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

```
60 passed in 58.75s
```

All tests run on a 4 GB RAM machine with no GPU. No model downloads. No expensive compute. This was a hard constraint — and it holds.

---

## Hardware This Was Built On

This isn't a disclaimer. It's a design constraint that shaped every decision.

| Component | Spec |
|-----------|------|
| OS | Windows 10 Pro 64-bit |
| CPU | Intel Core i3-5005U @ 2.00 GHz |
| RAM | 4 GB |
| GPU | None |
| CUDA | Not available |

Every line of code that touches memory was written with this machine in mind. Lazy loading, single-worker configs, batch size 1, demo mode fallbacks — none of that is incidental.

---

## What we're honest about

Some projects oversell what they've built. We'd rather be clear.

- **DP is a demonstration.** We add Gaussian noise and label it correctly. A formal (ε, δ) privacy guarantee needs rigorous calibration against a specific threat model — that's beyond a hackathon scope.
- **Secure aggregation is a simulation.** We model the additive masking concept. Production SecAgg needs secret sharing or homomorphic encryption.
- **The local model produces rough output.** `sshleifer/tiny-gpt2` has ~117K parameters. It's there for testing infrastructure, not impressing anyone with generation quality.
- **No real user data was used.** All datasets are synthetic, hand-crafted in Hindi, Marathi, and Tamil.

We believe a working, honest prototype is more valuable than an impressive demo that hides its seams.

---

## Demo Flow (5 minutes)

1. `streamlit run app.py` → open `http://localhost:8501`
2. **Home** — show the hardware profile, execution mode badge, live RAM/CPU usage
3. **Personalization** — pick Client 1 (Healthcare / Hindi), browse the synthetic dataset, generate a base and personalized response
4. Point out: 0 bytes of raw data transmitted
5. **Federation** — run the demo simulation, watch the loss curves update across 3 clients and 1 round
6. Walk through the FedAvg aggregation code block on screen
7. **Privacy Audit** — show the data flow diagram and the audit table
8. **Benchmarks** — show the adapter size vs full model bar chart (49K vs 85M parameters)
9. Wrap up: *"The laptop is the constrained edge client. The GPU cloud does the heavy lifting. The data boundary never moves."*

---

## References

- Hu, E. et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022.*
- Dettmers, T. et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). *NeurIPS 2023.*
- McMahan, B. et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). *AISTATS 2017.*
- Balle, B. et al. (2022). [Reconstructing Training Data with Informed Adversaries](https://arxiv.org/abs/2201.04845).
- [Flower Framework](https://flower.dev) — A Friendly Federated Learning Framework.

---

<div align="center">

*Built for a hackathon. Designed for the real world.*

**Personal data stays with the client.**
**Model knowledge travels as compressed weights.**
**That's the whole point.**

</div>
