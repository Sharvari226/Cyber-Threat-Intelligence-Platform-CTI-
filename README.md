# Cyber-Threat-Intelligence-Platform-CTI
A proactive defense system that predicts and neutralizes cyberattacks by analyzing global threat data and local network behavior in real-time.

# 🛡️ Cyber Threat Intelligence Platform
### Proactive Cyberattack Detection using Graph Neural Networks + Agentic AI

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![GCN](https://img.shields.io/badge/Model-GCN-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

A proactive cyber defense system that detects and responds to network attacks 
autonomously using **Graph Convolutional Networks (GCN)** and a **multi-agent 
AI pipeline**. Built on the UNSW-NB15 benchmark dataset, the system combines 
graph-based traffic modeling with LLM-powered agents that monitor, analyze, 
and act on threats in real time — without human intervention.

> **"Not just detection. Autonomous response."**

---

##  Key Features

-  **Graph-based traffic modeling** — network flows represented as nodes 
  and edges using NetworkX
- **GCN threat scoring** — Graph Convolutional Network classifies each 
  flow as Normal / Suspicious / High Risk
- **3 autonomous AI agents** — Watcher, Analyzer, Decider running in a 
  ReAct reasoning loop
-  **Autonomous alerting** — agents raise alerts and blacklist malicious 
  IPs without human input
-  **Self-monitoring** — agents recommend model retraining if accuracy drops
-  **Live Streamlit dashboard** — real-time graph visualization, alert 
  timeline, accuracy trends
-  **SQLite persistence** — all flows, alerts, and accuracy logs stored 
  in a local database
-  **Human override** — dashboard allows manual IP unblacklisting

---

##  System Architecture
```
UNSW-NB15 Dataset
       │
       ▼
┌─────────────────┐
│  Watcher Agent  │  ← monitors traffic, builds network graph
└────────┬────────┘
         │ flows + graph
         ▼
┌─────────────────┐
│ Analyzer Agent  │  ← runs GCN inference, assigns threat levels
└────────┬────────┘
         │ HIGH / SUSPICIOUS / NORMAL
         ▼
┌─────────────────┐
│  Decider Agent  │  ← raises alerts, blacklists IPs, suggests retraining
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     SQLite Database             │  ← stores flows, alerts, accuracy
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Streamlit Dashboard         │  ← live visualization + human override
└─────────────────────────────────┘
```

---

# Development Approach — Spiral SDLC

This project follows the **Spiral model** (Agile + Risk-driven), built in 4 loops:

# Loop 1 — GCN Foundation
- Load and preprocess UNSW-NB15 dataset
- Build network graph using NetworkX
- Train Graph Convolutional Network (PyTorch Geometric)
- Achieved **90%+ accuracy** on test set
- Deliverable: Working GCN model

# Loop 2 — Real-Time Simulation & Dashboard
- Simulate live traffic by processing dataset rows sequentially
- Dynamic graph updates with NetworkX
- Rule-based alerting (score > threshold → log warning)
- Streamlit dashboard with graph visualization
- SQLite database for history
- Deliverable: Dashboard prototype + simulated real-time detection

# Loop 3 — Agentic AI Layer
- 3 LangChain ReAct agents (Watcher, Analyzer, Decider)
- Agents communicate through shared state blackboard
- Autonomous alert raising, IP blacklisting, retrain suggestions
- Deliverable: Fully autonomous threat detection pipeline

# Loop 4 — Testing & Final Polish *(in progress)*
- Targeted attack replay (exploit/backdoor scenarios)
- False positive measurement
- Human override in dashboard
- Final comparison vs traditional ML
- Deliverable: Complete system + demo

---

# Agent Design

### WatcherAgent
- **Job:** Monitor incoming traffic, update network graph
- **Tools:** `ingest_traffic_batch`, `get_graph_status`
- **Reasoning:** Samples flows from dataset → adds to NetworkX graph → 
  reports protocols and patterns

### AnalyzerAgent
- **Job:** Run GCN inference, interpret predictions, assign threat levels
- **Tools:** `run_gcn_inference`, `evaluate_model_accuracy`
- **Reasoning:** Scores each flow → categorizes as HIGH/SUSPICIOUS/NORMAL → 
  checks model health

### DeciderAgent
- **Job:** Take autonomous action based on threat intelligence
- **Tools:** `raise_alert`, `blacklist_ip`, `recommend_retraining`, 
  `get_alerts_summary`
- **Reasoning:** HIGH flow → alert + blacklist IP | SUSPICIOUS → alert only | 
  accuracy drop → retrain flag

### ReAct Reasoning Loop
```
Thought  →  Action  →  Observation  →  Thought  →  ...  →  Final Answer
```
Each agent thinks step-by-step, picks a tool, reads the result, 
and decides the next action. This is printed live in the terminal.

---

# Project Structure
```
pbl/
├── app.py                          # Unified app (Loop 1+2+3 connected)
├── gcn.ipynb                       # Loop 1 — GCN training notebook
├── threat_intel.db                 # SQLite database (auto-created)
├── UNSW_NB15_training-set.csv      # Training data
├── UNSW_NB15_testing-set.csv       # Testing data
├── NUSW-NB15_features.csv          # Feature descriptions
├── agents/
│   ├── run_simulation.py           # Loop 3 — standalone agent simulation
│   └── mock_llm.py                 # Demo mode (no API key needed)
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Dataset | UNSW-NB15 |
| Graph Library | NetworkX |
| GCN Model | PyTorch Geometric / RandomForest proxy |
| Agent Framework | LangChain (ReAct) |
| LLM | Llama 3.1 via Groq API |
| Dashboard | Streamlit |
| Database | SQLite |
| ML Utilities | scikit-learn, pandas, numpy |
| Visualization | matplotlib |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/sharvari226/cyber-threat-intelligence.git
cd cyber-threat-intelligence
```

### 2. Install dependencies
```bash
pip install streamlit matplotlib networkx langchain langchain-classic \
            langchain-groq langchain-community groq scikit-learn \
            pandas numpy python-dotenv
```

### 3. Add your Groq API key

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
Or directly in `app.py`:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### 4. Add the dataset

Download UNSW-NB15 from the 
[official source](https://research.unsw.edu.au/projects/unsw-nb15-dataset) 
and place these files in the project root:
```
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
NUSW-NB15_features.csv
```

### 5. Run the dashboard
```bash
# Windows
python -m streamlit run app.py

# Mac/Linux
streamlit run app.py
```

Opens at **http://localhost:8501**


# 🚀 Usage

### Full Dashboard (recommended)
```bash
python -m streamlit run app.py
```
1. Dashboard opens in browser
2. Use the sidebar slider to set number of cycles (1–5)
3. Click **▶ Run Agent Simulation**
4. Watch agents reason live — graph updates, alerts appear, IPs get blacklisted
5. Use **Human Override** in sidebar to manually unblacklist an IP

# Standalone Agent Simulation (terminal only)
```bash
# With Groq API key
python agents/run_simulation.py

# Demo mode — no API key needed
python agents/run_simulation.py --mock

# Custom cycles
python agents/run_simulation.py --mock --cycles 5

# GCN Training Notebook
Open `gcn.ipynb` in Jupyter or Google Colab to train the full 
PyTorch Geometric GCN model.


# Results

| Metric | Value |
|---|---|
| GCN Test Accuracy | 90.57% |
| Alerts raised (3 cycles) | 11 |
| IPs blacklisted (3 cycles) | 10 |
| Avg model accuracy | 90.57% |
| Retrain triggered | No (healthy) |
| Target accuracy (spec) | 85–90% |
| Achieved |  Above target |


# Database Schema
```sql
-- Security alerts
alerts (id, timestamp, alert_id, flow_id, threat_level, src_ip, reason)

-- All analysed flows
flows (id, timestamp, flow_id, src_ip, dst_ip, proto, 
       attack_cat, risk_score, threat_level)

-- Model accuracy history
accuracy_log (id, timestamp, accuracy, cycle)

# Known Limitations

- GCN in `app.py` uses a **RandomForest proxy** due to PyTorch disk space 
  constraints. The full GCN is in `gcn.ipynb` and achieves the same accuracy.
- Groq free tier has a **100k token/day limit** for large models. 
  Use `llama-3.1-8b-instant` (500k/day) for extended runs.
- IPs are simulated (randomly generated) since UNSW-NB15 anonymizes real IPs.

# Future Scope

- [ ] Integrate real PyTorch Geometric GCN into the live pipeline
- [ ] Connect to real network traffic via packet capture (scapy/pyshark)
- [ ] Add email/SMS notifications for HIGH alerts
- [ ] Deploy dashboard to cloud (AWS/GCP/Heroku)
- [ ] Add more attack categories from newer datasets (CIC-IDS2018)
- [ ] Multi-model ensemble (GCN + LSTM for temporal patterns)

> Built as part of Project Based Learning (PBL) — 
> demonstrating modern AI in cybersecurity through 
> graph learning and agentic autonomy.
