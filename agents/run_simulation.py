"""
run_simulation.py  —  Agentic AI Layer for GCN-based Network Intrusion Detection
==================================================================================
Three LangChain ReAct agents running in sequence each cycle:

  WatcherAgent  → ingests live traffic, updates network graph
  AnalyzerAgent → runs GCN inference, assigns threat levels, checks accuracy
  DeciderAgent  → raises alerts, blacklists IPs, recommends retraining

Usage
-----
# No API key needed (deterministic mock demo):
    python run_simulation.py --mock

# Groq cloud LLM (free at console.groq.com):
    export GROQ_API_KEY="gsk_..."
    python run_simulation.py

# Local Ollama (run `ollama serve` + `ollama pull llama3` first):
    python run_simulation.py --ollama

# Custom number of cycles:
    python run_simulation.py --mock --cycles 5
"""

import os, sys, json, time, random, threading, argparse
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mock",   action="store_true", help="Use mock LLM")
parser.add_argument("--ollama", action="store_true", help="Use local Ollama")
parser.add_argument("--cycles", type=int, default=3)
args = parser.parse_args()

GROQ_API_KEY = "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"
USE_MOCK     = args.mock or (not GROQ_API_KEY and not args.ollama)
USE_OLLAMA   = args.ollama

if USE_MOCK:
    print("ℹ️  MockLLM demo mode — export GROQ_API_KEY=... to use a real LLM.\n")
elif USE_OLLAMA:
    print("ℹ️  Using local Ollama (llama3).\n")
else:
    print("✅  Using Groq (llama3-70b-8192).\n")

# ── LangChain imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────
RETRAIN_THRESHOLD = 0.80
ALERT_COOLDOWN    = 2   # seconds between alerts
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, "UNSW_NB15_training-set.csv")
TEST_CSV  = os.path.join(BASE_DIR, "UNSW_NB15_testing-set.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GCN Proxy Model
#     Mirrors the interface of the GCN in gcn.ipynb.
#     To use the real GCN: replace predict_flow() with a torch_geometric call.
# ══════════════════════════════════════════════════════════════════════════════
class GCNProxy:
    CAT_COLS = ["proto", "service", "state"]

    def __init__(self):
        self.model = self.encoder = self.scaler = self.feat_cols = None
        self.trained = False

    def fit(self, path: str, sample: int = 20_000):
        print(f"[GCNProxy] Sampling {sample} training rows …")
        df = pd.read_csv(path).sample(min(sample, 82_332), random_state=42)
        df = self._prep(df, fit=True)
        X, y = df[self.feat_cols].values, df["label"].values
        print(f"[GCNProxy] Training RandomForest on {len(X)} samples …")
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        self.trained = True
        print("[GCNProxy] ✅  Ready.\n")

    def evaluate(self, df_sample: pd.DataFrame) -> float:
        y_true = df_sample["label"].values          # save before _prep drops it
        df     = self._prep(df_sample.copy(), fit=False)
        return round(float(accuracy_score(y_true,
                    self.model.predict(df[self.feat_cols].values))), 4)

    def predict_flow(self, flow: dict) -> dict:
        """
        Single-flow inference.
        ── With torch installed, replace body with: ──
            out  = model(graph_data)
            prob = torch.softmax(out, dim=1)[node_idx][1].item()
        """
        row = pd.DataFrame([flow])
        row.drop(columns=[c for c in ("_flow_id","_src_ip","_dst_ip")
                          if c in row.columns], inplace=True)
        row  = self._prep(row, fit=False)
        prob = float(self.model.predict_proba(row[self.feat_cols].values)[0][1])
        lvl  = "HIGH" if prob > .80 else ("SUSPICIOUS" if prob > .50 else "NORMAL")
        return {"label": int(prob > .5), "risk_score": round(prob, 4), "threat_level": lvl}

    def _prep(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        if "id" in df.columns: df.drop(columns=["id"], inplace=True)
        for c in df.columns:
            if c in ("label", "attack_cat"): continue
            df[c] = df[c].fillna("unknown" if df[c].dtype == object else 0)
        cats = [c for c in self.CAT_COLS if c in df.columns]
        if fit:
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
            df[cats] = self.encoder.fit_transform(df[cats])
        elif cats:
            df[cats] = self.encoder.transform(df[cats])
        drop = [c for c in ("label","attack_cat") if c in df.columns]
        if fit:
            self.feat_cols = [c for c in df.columns if c not in drop]
            self.scaler    = StandardScaler()
            df[self.feat_cols] = self.scaler.fit_transform(df[self.feat_cols])
        else:
            df[self.feat_cols] = self.scaler.transform(df[self.feat_cols])
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Shared State  —  the agent blackboard
# ══════════════════════════════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.graph_nodes   = {}          # flow_id → flow dict
        self.graph_edges   = []          # (src_ip, dst_ip) list
        self.pending_flows = deque()     # awaiting GCN inference
        self.alerts        = []
        self.ip_blacklist  = set()
        self.accuracy_log  = deque(maxlen=30)
        self.retrain_flag  = False
        self.flow_counter  = 0
        self.last_alert_ts = 0.0

    def summary(self) -> str:
        with self.lock:
            return json.dumps({
                "graph_nodes":     len(self.graph_nodes),
                "graph_edges":     len(self.graph_edges),
                "pending_flows":   len(self.pending_flows),
                "total_flows":     self.flow_counter,
                "alerts_raised":   len(self.alerts),
                "blacklisted_ips": sorted(self.ip_blacklist),
                "avg_accuracy":    (round(float(np.mean(list(self.accuracy_log))), 4)
                                    if self.accuracy_log else None),
                "retrain_needed":  self.retrain_flag,
            }, indent=2)


# Module-level singletons (tools reference these via closure)
MODEL   = GCNProxy()
STATE   = SharedState()
TEST_DF = None   # populated in main()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Tools  —  each agent only sees its own subset
# ══════════════════════════════════════════════════════════════════════════════

# ── Watcher tools ─────────────────────────────────────────────────────────────

@tool
def ingest_traffic_batch(n: str = "8") -> str:
    """
    Sample N traffic flows from the live stream, add to graph + analysis queue.
    Returns JSON: ingested_count, graph_state, sample_flows.
    Input: n as a plain number e.g. 8
    """
    global TEST_DF
    if TEST_DF is None:
        return json.dumps({"error": "Test dataset not loaded."})
    try:
        import re
        n = int(re.search(r'\d+', str(n)).group())
        sample = TEST_DF.sample(min(n, len(TEST_DF)))
    except Exception as e:
        return json.dumps({"error": str(e)})

    ingested = []
    with STATE.lock:
        for _, row in sample.iterrows():
            fid = f"flow_{STATE.flow_counter:05d}"
            STATE.flow_counter += 1
            f = row.to_dict()
            f["_flow_id"] = fid
            f["_src_ip"]  = (f"10.{random.randint(0,5)}."
                             f"{random.randint(0,255)}.{random.randint(1,254)}")
            f["_dst_ip"]  = (f"192.168.{random.randint(0,3)}."
                             f"{random.randint(1,100)}")
            STATE.graph_nodes[fid] = f
            STATE.graph_edges.append((f["_src_ip"], f["_dst_ip"]))
            STATE.pending_flows.append(f)
            ingested.append({
                "flow_id":    fid,
                "proto":      str(row.get("proto", "?")),
                "service":    str(row.get("service", "?")),
                "src_ip":     f["_src_ip"],
                "label":      int(row.get("label", -1)),
                "attack_cat": str(row.get("attack_cat", "Normal")),
            })

    return json.dumps({
        "ingested_count": len(ingested),
        "graph_state":    {"nodes": len(STATE.graph_nodes),
                           "edges": len(STATE.graph_edges)},
        "sample_flows":   ingested[:4],
    }, indent=2)


@tool
def get_graph_status() -> str:
    """Return current network graph state summary."""
    return STATE.summary()


# ── Analyzer tools ────────────────────────────────────────────────────────────

@tool
def run_gcn_inference(batch_size: str = "8") -> str:
    """
    Pull pending flows, run GCN inference, return risk scores + threat levels.
    HIGH > 0.80  |  SUSPICIOUS 0.50–0.80  |  NORMAL < 0.50
    Input: batch_size as a plain number e.g. 8
    """
    if not MODEL.trained:
        return json.dumps({"error": "Model not trained."})
    # clean input — LLM sometimes sends "batch_size = 8" or "8" or 8
    try:
        import re
        batch_size = int(re.search(r'\d+', str(batch_size)).group())
    except:
        batch_size = 8
    results    = []
    with STATE.lock:
        flows = [STATE.pending_flows.popleft()
                 for _ in range(min(batch_size, len(STATE.pending_flows)))]
    for f in flows:
        try:
            pred = MODEL.predict_flow(f)
            pred.update({"flow_id": f.get("_flow_id","?"),
                         "src_ip": f.get("_src_ip","?"),
                         "proto":  str(f.get("proto","?")),
                         "attack_cat": str(f.get("attack_cat","Normal"))})
        except Exception as e:
            pred = {"flow_id": f.get("_flow_id","?"), "error": str(e),
                    "threat_level": "UNKNOWN", "risk_score": -1}
        results.append(pred)

    high = [r for r in results if r.get("threat_level") == "HIGH"]
    sus  = [r for r in results if r.get("threat_level") == "SUSPICIOUS"]
    return json.dumps({
        "summary":         {"total": len(results), "HIGH": len(high),
                            "SUSPICIOUS": len(sus),
                            "NORMAL": len(results)-len(high)-len(sus)},
        "high_risk_flows":  high,
        "suspicious_flows": sus,
        "all_predictions":  results,
        "remaining_pending": len(STATE.pending_flows),
    }, indent=2)


@tool
def evaluate_model_accuracy() -> str:
    """
    Evaluate GCN accuracy on a held-out test batch.
    Sets retrain flag automatically if accuracy < threshold.
    """
    global TEST_DF
    if not MODEL.trained or TEST_DF is None:
        return json.dumps({"error": "Model/data not ready."})
    sample = TEST_DF.sample(min(2000, len(TEST_DF)))
    acc    = MODEL.evaluate(sample)
    with STATE.lock:
        STATE.accuracy_log.append(acc)
        if acc < RETRAIN_THRESHOLD:
            STATE.retrain_flag = True
    avg = round(float(np.mean(list(STATE.accuracy_log))), 4)
    return json.dumps({
        "current_accuracy":    acc,
        "average_accuracy":    avg,
        "retrain_threshold":   RETRAIN_THRESHOLD,
        "retrain_recommended": acc < RETRAIN_THRESHOLD,
        "note": ("⚠️ Below threshold — retrain_flag set."
                 if acc < RETRAIN_THRESHOLD else "✅ Accuracy healthy."),
    }, indent=2)


# ── Decider tools ─────────────────────────────────────────────────────────────

@tool
def raise_alert(input_json: str) -> str:
    """
    Raise a security alert for a HIGH or SUSPICIOUS flow.
    Input must be a JSON string with keys: flow_id, threat_level, reason, src_ip
    Example: {"flow_id": "flow_00001", "threat_level": "HIGH", "reason": "DoS pattern", "src_ip": "10.0.1.5"}
    """
    try:
        import re, json as _json
        # strip any markdown code fences the LLM might add
        clean = re.sub(r"```json|```", "", input_json).strip()
        # handle key=value style input from LLM
        if "{" not in clean:
            parts = dict(re.findall(r'(\w+)\s*=\s*["\']?([^,\'"]+)["\']?', clean))
            clean = _json.dumps(parts)
        data = _json.loads(clean)
    except Exception as e:
        return json.dumps({"error": f"Could not parse input: {e}", "raw": input_json})

    flow_id      = str(data.get("flow_id", "unknown"))
    threat_level = str(data.get("threat_level", "UNKNOWN"))
    reason       = str(data.get("reason", "no reason given"))
    src_ip       = str(data.get("src_ip", "unknown"))

    now = time.time()
    with STATE.lock:
        if now - STATE.last_alert_ts < ALERT_COOLDOWN:
            return json.dumps({"status": "THROTTLED"})
        alert = {
            "alert_id":     f"ALERT-{len(STATE.alerts)+1:04d}",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "flow_id":      flow_id,
            "threat_level": threat_level,
            "src_ip":       src_ip,
            "reason":       reason,
        }
        STATE.alerts.append(alert)
        STATE.last_alert_ts = now
    print(f"\n🚨  [{alert['alert_id']}] {threat_level} | {src_ip} | {reason}")
    return json.dumps({"status": "ALERT_RAISED", "alert": alert})


@tool
def blacklist_ip(input_json: str) -> str:
    """
    Blacklist a source IP address.
    Input must be a JSON string with keys: ip, reason
    Example: {"ip": "10.0.1.5", "reason": "HIGH risk DoS detected"}
    """
    try:
        import re, json as _json
        clean = re.sub(r"```json|```", "", input_json).strip()
        if "{" not in clean:
            parts = dict(re.findall(r'(\w+)\s*=\s*["\']?([^,\'"]+)["\']?', clean))
            clean = _json.dumps(parts)
        data = _json.loads(clean)
    except Exception as e:
        return json.dumps({"error": f"Could not parse input: {e}", "raw": input_json})

    ip     = str(data.get("ip", "unknown"))
    reason = str(data.get("reason", "no reason given"))

    with STATE.lock:
        STATE.ip_blacklist.add(ip)
        total = len(STATE.ip_blacklist)
    print(f"🚫  BLACKLIST: {ip} — {reason}  (total: {total})")
    return json.dumps({"status": "BLACKLISTED", "ip": ip,
                       "reason": reason, "total_blacklisted": total})


@tool
def recommend_retraining(reason: str = "accuracy degradation") -> str:
    """
    Recommend GCN model retraining due to accuracy drop or concept drift.
    Input: a plain text reason string.
    Example: accuracy dropped below 0.80 threshold
    """
    # clean up if LLM passes reason="..." with quotes
    reason = str(reason).strip().strip('"').strip("'")
    with STATE.lock:
        STATE.retrain_flag = True
        avg = (round(float(np.mean(list(STATE.accuracy_log))), 4)
               if STATE.accuracy_log else None)
    msg = (f"⚙️  RETRAIN RECOMMENDED | reason={reason} | "
           f"avg_acc={avg} | threshold={RETRAIN_THRESHOLD}")
    print(f"\n{msg}")
    return json.dumps({"status": "RETRAIN_RECOMMENDED", "reason": reason,
                       "avg_accuracy": avg, "threshold": RETRAIN_THRESHOLD})


@tool
def get_alerts_summary(input: str = "") -> str:
    """
    Return full summary of all alerts, blacklisted IPs, and system status.
    No input needed — just call it.
    """
    with STATE.lock:
        return json.dumps({
            "total_alerts":      len(STATE.alerts),
            "blacklisted_ips":   sorted(STATE.ip_blacklist),
            "total_blacklisted": len(STATE.ip_blacklist),
            "recent_alerts":     STATE.alerts[-5:],
            "retrain_flag":      STATE.retrain_flag,
            "avg_accuracy":      (round(float(np.mean(list(STATE.accuracy_log))), 4)
                                  if STATE.accuracy_log else None),
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LLM factory
# ══════════════════════════════════════════════════════════════════════════════

def get_llm():
    if USE_MOCK:
        from mock_llm import MockChatLLM
        return MockChatLLM()
    if USE_OLLAMA:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3", temperature=0.1)
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile",
                temperature=0.1, max_tokens=2048)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ReAct agent builder
# ══════════════════════════════════════════════════════════════════════════════

REACT_TEMPLATE = """\
You are {agent_name}. {agent_role}

Available tools:
{tools}

STRICT FORMAT — use exactly this every step:

Question: the task
Thought: what to do next
Action: one of [{tool_names}]
Action Input: input to the tool
Observation: result
... (repeat Thought/Action/Observation as needed)
Thought: I have everything I need
Final Answer: your conclusion and next recommendation

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_agent(name: str, role: str, tools: list) -> AgentExecutor:
    prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
        agent_name=name, agent_role=role)
    agent  = create_react_agent(llm=get_llm(), tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True,
                         max_iterations=12, handle_parsing_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Orchestrator  —  runs the 3-agent pipeline each cycle
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self):
        print("[Orchestrator] Building agents …")
        self.watcher = build_agent(
            "WatcherAgent",
            "Monitor incoming network traffic. Ingest flows into the graph. "
            "Report graph state and summarise observed protocols.",
            [ingest_traffic_batch, get_graph_status])

        self.analyzer = build_agent(
            "AnalyzerAgent",
            "Run GCN inference on pending flows. Interpret risk scores. "
            "Evaluate model accuracy. Report all HIGH and SUSPICIOUS flows.",
            [run_gcn_inference, evaluate_model_accuracy])

        self.decider = build_agent(
            "DeciderAgent",
            "Take autonomous action on threat intelligence. "
            "raise_alert for every HIGH flow. blacklist_ip for every HIGH source. "
            "raise_alert for every SUSPICIOUS flow. "
            "If retrain recommended → call recommend_retraining. "
            "Always end with get_alerts_summary.",
            [raise_alert, blacklist_ip, recommend_retraining, get_alerts_summary])

        print("[Orchestrator] ✅  All three agents ready.\n")

    def run_cycle(self, cycle: int) -> dict:
        bar = "─" * 68
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        print(f"\n{bar}\n  CYCLE {cycle}  |  {ts}\n{bar}\n")

        # ── Step 1: Watcher ───────────────────────────────────────────────────
        print("▶  [1/3] WATCHER AGENT\n")
        w = self.watcher.invoke({"input":
            "Ingest 8 new network traffic flows. "
            "Retrieve the graph status. "
            "Summarise protocols observed and flag any unusual patterns."
        })["output"]
        print(f"\n✅  Watcher concluded:\n{w}\n")

        # ── Step 2: Analyzer ──────────────────────────────────────────────────
        print("▶  [2/3] ANALYZER AGENT\n")
        a = self.analyzer.invoke({"input":
            f"Watcher report: {w[:350]}\n\n"
            "Run GCN inference (batch_size=8). "
            "Evaluate model accuracy. "
            "List every HIGH and SUSPICIOUS flow with flow_id, src_ip, risk_score. "
            "State clearly whether retraining is recommended."
        })["output"]
        print(f"\n✅  Analyzer concluded:\n{a}\n")

        # ── Step 3: Decider ───────────────────────────────────────────────────
        print("▶  [3/3] DECIDER AGENT\n")
        d = self.decider.invoke({"input":
            f"Analyzer report: {a[:500]}\n\n"
            "Step-by-step action checklist:\n"
            "1. raise_alert for every HIGH-risk flow (with flow_id, src_ip, reason)\n"
            "2. blacklist_ip for every HIGH source IP\n"
            "3. raise_alert for every SUSPICIOUS flow\n"
            "4. If retraining recommended → call recommend_retraining\n"
            "5. Call get_alerts_summary to confirm all actions taken"
        })["output"]
        print(f"\n✅  Decider concluded:\n{d}\n")

        return {"cycle": cycle, "watcher": w, "analyzer": a, "decider": d}

    def run_simulation(self, cycles: int = 3) -> list:
        mode = "MOCK" if USE_MOCK else ("OLLAMA" if USE_OLLAMA else "GROQ")
        print("\n" + "═" * 68)
        print(f"  AGENTIC GCN INTRUSION DETECTION SYSTEM")
        print(f"  Cycles: {cycles}  |  LLM: {mode}")
        print("═" * 68 + "\n")

        results = []
        for i in range(1, cycles + 1):
            results.append(self.run_cycle(i))
            if i < cycles:
                time.sleep(0.5)

        print("\n" + "═" * 68)
        print("  SIMULATION COMPLETE — FINAL SYSTEM STATE")
        print("═" * 68)
        print(STATE.summary())
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global TEST_DF

    # Train model on a balanced random sample of the full training set
    MODEL.fit(TRAIN_CSV, sample=20_000)

    # Load test data using .sample() so we get balanced label distribution
    print("[Data] Loading test dataset …")
    TEST_DF = pd.read_csv(TEST_CSV).sample(15_000, random_state=42)
    print(f"[Data] ✅  {len(TEST_DF)} test rows ready "
          f"(label dist: {TEST_DF['label'].value_counts().to_dict()})\n")

    Orchestrator().run_simulation(cycles=args.cycles)


if __name__ == "__main__":
    main()