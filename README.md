# PRIVFEDQLORA

**Privacy-Preserving On-Device Personalization of LLMs via Federated QLoRA**

> A hackathon research prototype demonstrating how large language models can be personalized using private local data without transmitting that data to any server.

---

## Problem

Modern AI assistants improve by learning from user data. But collecting raw personal conversations, health records, or financial information on central servers creates serious privacy risks. Users in healthcare, education, and financial domains in India — communicating in Hindi, Marathi, and Tamil — deserve personalized AI assistance without surrendering their data.

## Motivation

**Federated Learning + LoRA = Privacy + Efficiency**

- Federated Learning (FL) allows models to be trained across many devices without sharing raw data.
- LoRA (Low-Rank Adaptation) reduces the trainable parameter count from hundreds of millions to tens of thousands.
- QLoRA adds 4-bit quantization, enabling large model fine-tuning on GPU-constrained hardware.
- Together, these techniques let a model learn from your private data on your device, then share only a tiny compressed adapter — not the data itself.

---

## Architecture

```
User Device (4 GB RAM, No GPU — e.g., Intel Core i3)
┌──────────────────────────────────────────────────────────────┐
│  🔐 Synthetic Private Local Data (Healthcare / Education /   │
│      Financial — Hindi / Marathi / Tamil)                    │
│  📊 Streamlit Dashboard                                       │
│  🔍 Hardware Monitoring                                       │
│  🎯 DEMO / LIGHTWEIGHT Inference                              │
│  🔒 Privacy Audit                                             │
│  📈 Benchmarks                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │  Only LoRA Adapter Weights (~192 KB)
                        │  NOT raw data
                        ▼
Cloud GPU Environment
┌──────────────────────────────────────────────────────────────┐
│  🏋️  QLoRA Training (4-bit NF4 + LoRA)                       │
│  🌐 3-Client Flower (flwr) Federated Simulation             │
│  📦 FedAvg Aggregation                                        │
│  💾 Adapter Export                                           │
└──────────────────────────────────────────────────────────────┘
```

### Execution Modes

| Mode | Where | Triggers |
|------|--------|----------|
| `DEMO` | Local laptop | RAM < 2.5 GB available, or model load fails |
| `LIGHTWEIGHT_LOCAL` | Local laptop | RAM ≥ 2.5 GB, no CUDA |
| `CLOUD_TRAINING` | Cloud GPU | CUDA available |

---

## Why On-Device?

1. **Privacy**: Raw data (medical queries, financial records, student notes) never leaves the device.
2. **Compliance**: GDPR, DPDP Act (India) — data minimization principle.
3. **Personalisation at the edge**: Models adapt to individual users without centralised data collection.
4. **Trust**: Users retain full control over their data.

## Why LoRA?

Full fine-tuning of a 1B+ parameter model requires tens of GB of GPU RAM. LoRA adds a small trainable matrix decomposition to selected layers, reducing trainable parameters by >99% while retaining most adaptation quality.

| | Full Fine-Tuning | LoRA (r=4) |
|---|---|---|
| Parameters updated | ~85M | ~49K |
| Communication payload | ~340 MB | ~192 KB |
| RAM required (train) | 32+ GB | 6–8 GB |

## Why QLoRA?

QLoRA applies 4-bit NF4 quantization to the frozen base model weights, allowing large models (7B, 13B) to fit in 6–8 GB GPU RAM during training. Combined with LoRA, it makes fine-tuning accessible on a single consumer GPU.

## Why Federated Learning?

Multiple clients (healthcare clinic, school, bank) each have private domain-specific data. FL allows them to collectively improve a shared model without pooling data:

1. Global adapter distributed to all clients
2. Each client trains locally on private data
3. Only adapter updates (not data) are returned
4. FedAvg aggregates updates
5. Improved global adapter returned

---

## Privacy Boundary

| What | Transmitted? | Status |
|------|-------------|--------|
| Raw personal data | **NO — 0 bytes** | IMPLEMENTED |
| Local training dataset | **NO** | IMPLEMENTED |
| LoRA adapter update | **YES** — ~192 KB | IMPLEMENTED |
| Update clipping (L2) | Applied before transmission | IMPLEMENTED |
| Gaussian DP noise | Added to updates | DEMONSTRATION |
| Secure aggregation | Additive masking simulation | SIMULATION |
| Formal DP proof | Not provided | NOT IMPLEMENTED |
| Cryptographic SecAgg | Not provided | NOT IMPLEMENTED |

---

## Hardware Limitations

This prototype targets:

- **OS:** Windows 10 Pro 64-bit
- **CPU:** Intel Core i3-5005U @ 2.00 GHz
- **RAM:** 4 GB
- **GPU:** None
- **CUDA:** Not available

**Local machine is responsible for:**
- Streamlit interface
- Synthetic data generation
- Hardware monitoring
- Privacy audit visualization
- DEMO mode inference
- Federated simulation orchestration (display only)

**Cloud GPU is responsible for:**
- QLoRA training (4-bit NF4)
- 3-client Flower simulation
- Model evaluation
- Adapter export

---

## Project Structure

```
PRIVFEDQLORA/
├── app.py                    # Streamlit dashboard entry point
├── config.py                 # Environment-driven configuration
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── core/
│   ├── hardware.py           # Hardware detection, execution mode
│   ├── logger.py             # Structured logging
│   └── exceptions.py        # Custom exceptions
├── data/
│   ├── synthetic_data.py     # 3-client synthetic datasets
│   ├── dataset.py            # LocalDataset wrapper
│   └── preprocessing.py     # Tokenization helpers
├── model/
│   ├── loader.py             # Lazy model loading with RAM gating
│   ├── lora.py               # LoRA config and application
│   ├── inference.py          # Generate responses (real + demo)
│   └── adapter.py            # Save/load/inspect adapter files
├── federated/
│   ├── client.py             # Flower NumPyClient + DemoFlowerClient
│   ├── server.py             # Flower server configuration
│   ├── strategy.py           # FedAvg strategy builder
│   └── simulation.py        # Demo + real simulation orchestration
├── privacy/
│   ├── clipping.py           # L2 norm update clipping
│   ├── differential_privacy.py  # Gaussian DP mechanism (demo)
│   ├── secure_aggregation.py    # Additive masking simulation
│   └── audit.py              # Privacy audit report
├── benchmark/
│   └── metrics.py            # RAM, CPU, latency, adapter size
├── ui/
│   ├── home.py               # Dashboard home page
│   ├── personalization.py    # Synthetic data + inference demo
│   ├── federation.py         # FL round control + visualization
│   ├── privacy.py            # Privacy audit dashboard
│   └── benchmarks.py        # Metrics display
├── storage/
│   ├── adapters/             # Saved adapter files
│   ├── datasets/             # Generated synthetic datasets
│   ├── results/              # Federated training results
│   └── logs/                 # Application logs
├── cloud/
│   ├── train_qlora.py        # Cloud QLoRA training script
│   ├── run_federated.py      # Cloud Flower simulation
│   └── export_adapter.py    # Adapter export utility
└── tests/
    ├── test_hardware.py
    ├── test_data.py
    ├── test_privacy.py
    └── test_smoke.py
```

---

## Installation

### Prerequisites

- Python 3.11
- pip

### Steps

```bash
cd PRIVFEDQLORA

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Environment Configuration

```bash
copy .env.example .env
```

Edit `.env` to configure:
- `MODEL_MODE` — `lightweight` (local small model) or `cloud`
- `LOCAL_MODEL_ID` — small model for local demo (default: `sshleifer/tiny-gpt2`)
- `CLOUD_MODEL_ID` — larger model for cloud training
- `HF_TOKEN` — HuggingFace token if accessing gated models
- `DP_ENABLED` — enable/disable differential privacy demonstration

---

## Running Locally

### Start the Streamlit Dashboard

```bash
cd PRIVFEDQLORA
streamlit run app.py
```

The application will:
1. Detect hardware automatically
2. Determine execution mode (DEMO / LIGHTWEIGHT_LOCAL)
3. Display hardware metrics on the Home page
4. Allow synthetic data viewing and personalization demo
5. Show federated workflow visualization
6. Display privacy audit
7. Show benchmark metrics

> On a 4 GB RAM machine without GPU, the app runs in **DEMO mode** — all outputs are clearly labelled as simulated.

---

## Running Cloud Training

**Requires: GPU machine with CUDA, PyTorch, bitsandbytes, peft, transformers**

### Train a single client with QLoRA

```bash
cd PRIVFEDQLORA
python cloud/train_qlora.py \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --client-id client_1 \
    --output-dir storage/adapters \
    --lora-r 4 \
    --lora-alpha 16
```

### Export an adapter

```bash
python cloud/export_adapter.py \
    --adapter-dir storage/adapters/client_1 \
    --output-dir storage/adapters/exported \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Or create a demo record (no GPU needed):
python cloud/export_adapter.py --demo --output-dir storage/adapters
```

---

## Running Federated Simulation

**Requires: GPU machine with CUDA, Flower (flwr)**

```bash
cd PRIVFEDQLORA
python cloud/run_federated.py \
    --num-rounds 1 \
    --output-dir storage/results
```

This runs a 3-client Flower simulation with:
- Client 1: Healthcare / Hindi
- Client 2: Education / Marathi
- Client 3: Financial Literacy / Tamil

Results are saved to `storage/results/federated_results.json`.

---

## Running Tests

```bash
cd PRIVFEDQLORA
python -m pytest tests/ -v --tb=short
```

Tests run without GPU, without downloading large models, and within 4 GB RAM.

---

## Known Limitations

1. **No real QLoRA on local machine** — 4 GB RAM cannot run 4-bit training. Cloud environment required.
2. **Flower simulation is cloud-only** — Local DEMO mode simulates FL with toy parameter vectors.
3. **DP is demonstration only** — Gaussian noise is added but formal (ε, δ) guarantees require rigorous calibration and auditing.
4. **Secure aggregation is simulation** — Additive masking simulates the concept; production requires cryptographic protocols.
5. **Tiny local model** — `sshleifer/tiny-gpt2` (~117K parameters) produces low-quality text; it exists for local testing only.
6. **Synthetic data only** — No real user data is used or required.

---

## Hackathon Demo Script

1. Open `http://localhost:8501` after `streamlit run app.py`
2. **Home** → Show hardware profile, RAM usage, execution mode
3. **Personalization** → Select Client 1 (Healthcare/Hindi), show synthetic data, run inference demo
4. Show privacy reminder: 0 bytes raw data transmitted
5. **Federation** → Show 3-client FedAvg architecture, run demo simulation, show loss/accuracy charts
6. **Privacy Audit** → Show data flow diagram, audit table, mechanism details
7. **Benchmarks** → Show RAM gauge, adapter size vs full model bar chart, communication efficiency pie chart
8. Explain: laptop is the constrained client; GPU cloud handles QLoRA training and Flower orchestration

---

## Core Message

> **Personal data stays with the client. Model knowledge is collaboratively improved through parameter-efficient updates — not by sharing raw data.**

---

## References

- Hu et al. (2022) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023) — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)](https://arxiv.org/abs/1602.05629)
- Balle et al. (2022) — [Differentially Private Federated Learning](https://arxiv.org/abs/2206.01151)
- [Flower Framework](https://flower.dev) — Friendly Federated Learning
