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

## Problem Statement

### The Dependency Problem

India's AI adoption is accelerating — yet the country remains deeply dependent on externally hosted large language model (LLM) platforms such as ChatGPT, Gemini, DeepSeek, and Kimi. Every query sent to these services potentially exposes personally identifiable information (PII): a patient's symptom history in Hindi, a farmer's crop disease query in Tamil, a household's UPI transaction logs, or a student's learning difficulties in Marathi. This silent data exfiltration is not hypothetical — it is the default operating mode.

The problem is particularly acute in the Indian context, where:

- **Linguistic diversity** demands domain-specific adaptation across 22 scheduled languages. A single globally trained model trained predominantly on English and Mandarin corpora performs poorly on code-switched Hindi-English queries, agglutinative Tamil morphology, or the Devanagari script used across Hindi, Marathi, Konkani, and Sanskrit.
- **Use cases carry high sensitivity** — vernacular health consultations, agricultural advisory (where wrong advice destroys a season's crop), microfinance eligibility queries, and school examination guidance all involve information that users would not willingly broadcast to foreign cloud servers.
- **Infrastructure is heterogeneous and constrained.** The median Indian smartphone ships with 4–6 GB RAM, a mid-range ARM SoC (e.g., Snapdragon 680 / MediaTek Helio G85) with no dedicated AI accelerator beyond a basic NPU, and a 4G connection that averages 15–25 Mbps but is frequently interrupted. Mobile data costs, while declining, remain a meaningful constraint for rural and semi-urban users.
- **Regulatory pressure is increasing.** India's Digital Personal Data Protection (DPDP) Act 2023 introduces binding obligations on data fiduciaries regarding consent, purpose limitation, and cross-border data transfer — creating legal risk for applications that blindly forward user conversations to offshore inference endpoints.

### What LokLLM Proposes

LokLLM is a research prototype that demonstrates a **privacy-first, on-device personalization pipeline** for 2–7 billion parameter language models, designed to operate within the hardware envelope of Indian mid-range consumer devices. The system integrates three complementary technologies:

#### 1. Edge-Deployable Base Models (2–7B Parameters, 4-bit NF4 Quantization)

Full-precision 7B models require ~28 GB of GPU VRAM — far beyond any consumer smartphone. We target the **4-bit NF4 quantization** tier via QLoRA (Dettmers et al., 2023), which reduces a 7B model to approximately 3.5–4 GB of resident memory, making inference and adapter training feasible on devices with 6–8 GB RAM. For the most constrained 4 GB class (the modal specification for sub-₹12,000 devices), we additionally evaluate 1.1B parameter models (TinyLlama, Gemma-2 2B) that fit within a 2.5 GB footprint at 4-bit precision. We explicitly test and report inference latency (tokens/second), memory high-watermark, and energy consumption across both tiers.

| Model Tier | Params | Quantization | RAM Footprint | Target Device |
|---|---|---|---|---|
| Tier 1 | 1.1–2B | 4-bit NF4 | ~1.5–2.5 GB | 4 GB RAM, no GPU |
| Tier 2 | 7B | 4-bit NF4 | ~3.5–4 GB | 6–8 GB RAM, entry NPU |
| Tier 3 | 7B | 8-bit | ~7 GB | 8 GB RAM, mid-range GPU |

#### 2. QLoRA Fine-Tuning: Adapter Ranks, Target Modules, and Serving Strategy

We adapt the frozen quantized base model using **Low-Rank Adaptation (LoRA)** injected into the attention projection layers. The specific implementation choices:

- **Adapter ranks under evaluation:** r ∈ {8, 16, 64}. Lower ranks (r=8, ~49K trainable parameters) minimize communication overhead and memory; higher ranks (r=64) allow richer domain capture at the cost of ~392K parameters per adapter — still under 1.5 MB.
- **Target modules:** We inject LoRA into `q_proj` and `v_proj` by default (following Hu et al., 2022), and separately ablate full attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and all linear layers including `gate_proj`, `up_proj`, `down_proj` in MLP blocks. The tradeoff between adaptation quality and adapter size is reported explicitly.
- **Adapter serving without full model reload:** Trained adapters are stored as separate `.safetensors` files and merged at inference time using PEFT's `merge_and_unload()` pathway. This allows the base model to remain resident in memory while domain-specific adapters are hot-swapped — a critical efficiency requirement for multi-domain use on resource-constrained devices.

#### 3. Federated Learning Architecture: Aggregation, Client Selection, and Cold-Start

The federation layer enables multiple isolated clients (clinics, schools, cooperative banks) to collaboratively improve a shared adapter without exchanging raw data.

- **Aggregation protocol:** We implement **FedAvg** (McMahan et al., 2017) as the baseline and provide a hook for **FedProx** (Li et al., 2020), which adds a proximal regularization term (μ) to stabilize training under the high statistical heterogeneity (non-IID data distributions) typical of real-world Indian deployments. FedProx is particularly relevant when one client's dataset is 10× larger than another's — a realistic scenario when a district hospital participates alongside a rural health sub-centre.
- **Client selection under device heterogeneity:** Not all devices can complete a training round within a given time window. We model asynchronous participation with a minimum viable cohort (at least 2 of N clients must return updates per round) and use weighted averaging proportional to local dataset size. Stragglers are excluded from the current round but participate in the next.
- **Communication frequency:** We target **1–3 communication rounds** for the prototype, with each round transmitting adapter deltas of ~192 KB (r=4) to ~1.5 MB (r=64) — well within a single HTTPS POST even on a 2G fallback connection. For real deployment, we envision opportunistic synchronization over Wi-Fi at night to avoid mobile data costs.
- **Cold-start handling:** New users receive the current global adapter as a starting point, pre-loaded onto the device alongside the base model. Initial personalization is achieved via **few-shot prompting** (5–10 representative examples stored locally) until enough local interactions accumulate to trigger a fine-tuning round (threshold: ≥50 labelled examples). This hybrid approach avoids the cold-start degradation that plagues pure federated systems.

### Supported Indian Use Cases

We demonstrate the system across three client personas that reflect real unmet needs:

| Client | Domain | Language(s) | Illustrative Queries |
|--------|--------|-------------|---------------------|
| 🏥 Client 1 | Rural Healthcare | Hindi + code-switched English | Symptom triage, medication dosage queries, referral advice in Devanagari script |
| 📚 Client 2 | School Education | Marathi | Curriculum-aligned doubt resolution, exam preparation, teacher aide for government school syllabus |
| 💰 Client 3 | Financial Literacy | Tamil | UPI payment guidance, KYC query handling, microfinance eligibility, SHG record-keeping |

Future extensions target agricultural advisory (pest identification, MSP price queries in Kannada/Telugu) and vernacular legal aid (tenant rights, MNREGA entitlements).

### Threat Model and Privacy Guarantees

"Privacy-preserving" is not a monolithic claim. We are explicit about what we protect against and what we do not:

| Threat | In Scope? | Mechanism |
|--------|-----------|-----------|
| Honest-but-curious aggregation server observing raw training data | ✅ Yes | Data never leaves the device; only adapter deltas are transmitted |
| Gradient inversion / model inversion attacks reconstructing training samples from adapter updates | ⚠️ Partial | L2 norm clipping + Gaussian DP noise (demonstration; not formally calibrated) |
| Colluding clients attempting to reconstruct another client's data | 🔵 Simulated | Additive secret masking modelled; cryptographic SecAgg not implemented |
| Inference-time user query privacy (queries to the local model) | ✅ Yes | All inference is local; no query leaves the device |
| Membership inference against the global model | ❌ Out of scope | Formal DP auditing with Rényi DP accountant not included |

Our differential privacy implementation adds calibrated Gaussian noise with a nominal **ε = 8, δ = 10⁻⁵** budget (demonstration values). A production deployment would require formal composition analysis across rounds using a Rényi DP accountant (Mironov, 2017) and a privacy audit against a shadow-model attack. We document this gap clearly rather than hiding it.

### Evaluation Methodology

Beyond task accuracy, we measure what actually matters for edge deployment:

**Datasets:**
- Synthetic multilingual dialogue sets (hand-crafted; Hindi, Marathi, Tamil) — 30 samples per client for the prototype
- IndicNLP Corpus and Samanantar parallel corpora for multilingual robustness evaluation (planned)
- Synthetic financial dialogue based on public RBI financial literacy materials
- Ayushman Bharat scheme FAQs for health domain evaluation

**Metrics:**
| Category | Metric |
|----------|--------|
| Quality | ROUGE-L, BERTScore (multilingual), human preference |
| Efficiency | Inference latency (tokens/sec), memory high-watermark (MB), energy (mAh/query) |
| Communication | Adapter size (KB), rounds to convergence |
| Privacy | Membership inference AUC, gradient norm distribution, DP ε budget consumed |
| Utility-privacy tradeoff | Quality vs. ε curves at r ∈ {8, 16, 64} |

**Baselines:**
1. Zero-shot prompting of the untuned base model
2. Centralized fine-tuning on pooled data (upper bound; privacy-violating)
3. Local fine-tuning without federation (no knowledge sharing)
4. PRIVFEDQLORA (FedAvg / FedProx with DP)

### Deployment Narrative and Regulatory Compliance

**Model distribution:** The base model (1.1–7B, 4-bit quantized) is distributed as a one-time download (~2–4 GB) over Wi-Fi, analogous to an OTA system update. For devices where even this is impractical, we explore sideloading via USB from a local service centre — a realistic distribution channel in Tier 3/4 Indian towns. Adapter updates (~200 KB) are distributed over any connection including 2G.

**Connectivity-resilient synchronization:** The federation client implements an **offline-first** design. Local training proceeds regardless of connectivity. Adapter uploads and downloads are queued and executed when a Wi-Fi connection is detected, using resumable HTTP uploads to tolerate mid-transfer drops.

**DPDP Act 2023 compliance posture:** Since no personal data leaves the device, the system avoids most obligations that arise from data fiduciary status. The aggregation server processes only anonymized adapter weight deltas, not personal data — a distinction that substantially reduces regulatory exposure. We recommend a Data Protection Impact Assessment (DPIA) for any production deployment and note that the DPDP Act's consent framework applies to the local data collection step, not the federated aggregation step.

### Honest Limitations

Staying on-device means making real sacrifices. We acknowledge them directly:

- **No real-time web access.** The local model's knowledge is frozen at its training cut-off. Queries requiring current news, live prices, or today's weather cannot be answered accurately.
- **Reduced context window.** Memory constraints limit practical context to 512–2048 tokens on 4 GB devices, versus 32K–128K tokens available in cloud APIs. Long documents cannot be processed in a single pass.
- **State-of-the-art reasoning gap.** A 1.1–7B on-device model is meaningfully weaker than GPT-4-class systems on complex multi-step reasoning, mathematics, and code generation. This is the fundamental quality-privacy tradeoff.
- **Hybrid cloud escape hatch:** For queries that genuinely require capabilities beyond on-device scope (e.g., real-time stock prices, complex legal document drafting), we envision an **opt-in, user-triggered** hybrid mode that routes the query to a privacy-respecting cloud endpoint with explicit consent, stripping PII before transmission via a local anonymization filter. This preserves the privacy guarantee for the default case while offering an upgrade path.

We built this prototype to demonstrate that the correct architecture, applied honestly on real hardware, is more valuable than an impressive demo that obscures its constraints.

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
git clone https://github.com/tayade-aniket/LokLLM
cd LokLLM
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
