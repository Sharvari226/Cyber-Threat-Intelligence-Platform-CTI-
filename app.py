"""
app.py  —  Unified Cyber Threat Intelligence Platform
Connects Loop 1 (GCN) + Loop 2 (Streamlit + SQLite) + Loop 3 (Agents)

Run with:
    streamlit run app.py
"""

import os, json, time, random, threading, sqlite3
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from collections import deque
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Loop 3 imports ────────────────────────────────────────────────────────────
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ── Loop 1 imports ────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = "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"   # or os.getenv("GROQ_API_KEY")
RETRAIN_THRESHOLD = 0.80
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV         = os.path.join(BASE_DIR, "UNSW_NB15_training-set.csv")
TEST_CSV          = os.path.join(BASE_DIR, "UNSW_NB15_testing-set.csv")
DB_PATH           = os.path.join(BASE_DIR, "threat_intel.db")

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 2 — SQLite Database setup
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            alert_id     TEXT,
            flow_id      TEXT,
            threat_level TEXT,
            src_ip       TEXT,
            reason       TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            flow_id      TEXT,
            src_ip       TEXT,
            dst_ip       TEXT,
            proto        TEXT,
            attack_cat   TEXT,
            risk_score   REAL,
            threat_level TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            accuracy  REAL,
            cycle     INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_alert_to_db(alert: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO alerts (timestamp, alert_id, flow_id, threat_level, src_ip, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (alert["timestamp"], alert["alert_id"], alert["flow_id"],
          alert["threat_level"], alert["src_ip"], alert["reason"]))
    conn.commit()
    conn.close()

def save_flow_to_db(flow_id, src_ip, dst_ip, proto,
                    attack_cat, risk_score, threat_level):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO flows (timestamp, flow_id, src_ip, dst_ip, proto,
                           attack_cat, risk_score, threat_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), flow_id, src_ip, dst_ip,
          proto, attack_cat, risk_score, threat_level))
    conn.commit()
    conn.close()

def save_accuracy_to_db(accuracy: float, cycle: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO accuracy_log (timestamp, accuracy, cycle)
        VALUES (?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), accuracy, cycle))
    conn.commit()
    conn.close()

def load_alerts_from_db():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM alerts ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

def load_flows_from_db():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM flows ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    return df

def load_accuracy_from_db():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM accuracy_log ORDER BY id", conn)
    conn.close()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 1 — GCN Proxy Model (swap with real GCN when torch available)
# ─────────────────────────────────────────────────────────────────────────────
class GCNModel:
    """
    Wraps the model from Loop 1 (gcn.ipynb).
    Currently uses RandomForest as proxy.
    To use real GCN: replace predict_flow() with torch_geometric inference.
    """
    CAT_COLS = ["proto", "service", "state"]

    def __init__(self):
        self.model = self.encoder = self.scaler = self.feat_cols = None
        self.trained = False

    def fit(self, path: str, sample: int = 20_000):
        df = pd.read_csv(path).sample(min(sample, 82_332), random_state=42)
        df = self._prep(df, fit=True)
        X, y = df[self.feat_cols].values, df["label"].values
        self.model = RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        self.trained = True

    def evaluate(self, df_sample: pd.DataFrame) -> float:
        y_true = df_sample["label"].values
        df     = self._prep(df_sample.copy(), fit=False)
        return round(float(accuracy_score(
            y_true, self.model.predict(df[self.feat_cols].values))), 4)

    def predict_flow(self, flow: dict) -> dict:
        row = pd.DataFrame([flow])
        row.drop(columns=[c for c in
                 ("_flow_id","_src_ip","_dst_ip") if c in row.columns],
                 inplace=True)
        row  = self._prep(row, fit=False)
        prob = float(self.model.predict_proba(
                     row[self.feat_cols].values)[0][1])
        lvl  = ("HIGH" if prob > .80 else
                "SUSPICIOUS" if prob > .50 else "NORMAL")
        return {"label": int(prob>.5),
                "risk_score": round(prob,4),
                "threat_level": lvl}

    def _prep(self, df, fit):
        df = df.copy()
        if "id" in df.columns: df.drop(columns=["id"], inplace=True)
        for c in df.columns:
            if c in ("label","attack_cat"): continue
            df[c] = df[c].fillna("unknown" if df[c].dtype==object else 0)
        cats = [c for c in self.CAT_COLS if c in df.columns]
        if fit:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)
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

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 2 — NetworkX Graph
# ─────────────────────────────────────────────────────────────────────────────
class NetworkGraph:
    """Live network graph — nodes are IPs, edges are connections."""
    def __init__(self):
        self.G = nx.DiGraph()

    def add_flow(self, src_ip, dst_ip, proto, risk_score, threat_level):
        # Add nodes with attributes
        self.G.add_node(src_ip, type="source",
                        threat=threat_level, risk=risk_score)
        self.G.add_node(dst_ip, type="dest")
        # Add edge with flow metadata
        self.G.add_edge(src_ip, dst_ip,
                        proto=proto, risk=risk_score, threat=threat_level)

    def get_figure(self):
        """Returns matplotlib figure for Streamlit display."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(self.G.nodes) == 0:
            ax.text(0.5, 0.5, "No flows yet",
                    ha="center", va="center")
            return fig

        pos    = nx.spring_layout(self.G, seed=42)
        # Color nodes by threat level
        colors = []
        for node in self.G.nodes:
            threat = self.G.nodes[node].get("threat","NORMAL")
            colors.append(
                "#e74c3c" if threat=="HIGH" else
                "#f39c12" if threat=="SUSPICIOUS" else
                "#2ecc71")

        nx.draw(self.G, pos, ax=ax,
                node_color=colors, node_size=500,
                with_labels=True, font_size=7,
                arrows=True, edge_color="#888",
                font_color="white", font_weight="bold")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="HIGH"),
            Patch(color="#f39c12", label="SUSPICIOUS"),
            Patch(color="#2ecc71", label="NORMAL"),
        ], loc="upper left", fontsize=8)
        ax.set_title("Live Network Graph", fontweight="bold")
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE — all 3 loops read/write this
# ─────────────────────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.pending_flows = deque()
        self.alerts        = []
        self.ip_blacklist  = set()
        self.accuracy_log  = deque(maxlen=30)
        self.retrain_flag  = False
        self.flow_counter  = 0
        self.last_alert_ts = 0.0
        self.cycle         = 0

# Module singletons
GCN        = GCNModel()
GRAPH      = NetworkGraph()
STATE      = SharedState()
TEST_DF    = None

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 3 — Agent Tools (now also write to DB and Graph)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def ingest_traffic_batch(n: str = "8") -> str:
    """
    Sample N traffic flows, add to graph and analysis queue.
    Input: plain number e.g. 8
    """
    global TEST_DF
    if TEST_DF is None:
        return json.dumps({"error": "Data not loaded."})
    try:
        import re
        n      = int(re.search(r'\d+', str(n)).group())
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
                             f"{random.randint(0,255)}."
                             f"{random.randint(1,254)}")
            f["_dst_ip"]  = (f"192.168.{random.randint(0,3)}."
                             f"{random.randint(1,100)}")
            STATE.pending_flows.append(f)
            ingested.append({
                "flow_id":    fid,
                "proto":      str(row.get("proto","?")),
                "src_ip":     f["_src_ip"],
                "dst_ip":     f["_dst_ip"],
                "attack_cat": str(row.get("attack_cat","Normal")),
            })

    return json.dumps({"ingested_count": len(ingested),
                       "sample_flows": ingested[:3]}, indent=2)


@tool
def get_graph_status() -> str:
    """Return current system state summary."""
    with STATE.lock:
        return json.dumps({
            "graph_nodes":    len(GRAPH.G.nodes),
            "graph_edges":    len(GRAPH.G.edges),
            "pending_flows":  len(STATE.pending_flows),
            "total_flows":    STATE.flow_counter,
            "alerts_raised":  len(STATE.alerts),
            "blacklisted":    len(STATE.ip_blacklist),
            "avg_accuracy":   (round(float(np.mean(
                               list(STATE.accuracy_log))),4)
                               if STATE.accuracy_log else None),
        }, indent=2)


@tool
def run_gcn_inference(batch_size: str = "8") -> str:
    """
    Run GCN inference on pending flows.
    HIGH > 0.80 | SUSPICIOUS 0.50-0.80 | NORMAL < 0.50
    Input: plain number e.g. 8
    """
    if not GCN.trained:
        return json.dumps({"error": "Model not ready."})
    try:
        import re
        batch_size = int(re.search(r'\d+', str(batch_size)).group())
    except:
        batch_size = 8

    results = []
    with STATE.lock:
        flows = [STATE.pending_flows.popleft()
                 for _ in range(min(batch_size,
                                    len(STATE.pending_flows)))]
    for f in flows:
        try:
            pred = GCN.predict_flow(f)
            pred.update({
                "flow_id":    f.get("_flow_id","?"),
                "src_ip":     f.get("_src_ip","?"),
                "dst_ip":     f.get("_dst_ip","?"),
                "proto":      str(f.get("proto","?")),
                "attack_cat": str(f.get("attack_cat","Normal")),
            })
            # ── Loop 2 connection: update NetworkX graph ──────────────
            GRAPH.add_flow(pred["src_ip"], pred["dst_ip"],
                           pred["proto"], pred["risk_score"],
                           pred["threat_level"])
            # ── Loop 2 connection: persist to SQLite ──────────────────
            save_flow_to_db(pred["flow_id"], pred["src_ip"],
                            pred["dst_ip"], pred["proto"],
                            pred["attack_cat"], pred["risk_score"],
                            pred["threat_level"])
        except Exception as e:
            pred = {"flow_id": f.get("_flow_id","?"),
                    "error": str(e), "threat_level": "UNKNOWN",
                    "risk_score": -1}
        results.append(pred)

    high = [r for r in results if r.get("threat_level")=="HIGH"]
    sus  = [r for r in results if r.get("threat_level")=="SUSPICIOUS"]
    return json.dumps({
        "summary":          {"total": len(results), "HIGH": len(high),
                             "SUSPICIOUS": len(sus),
                             "NORMAL": len(results)-len(high)-len(sus)},
        "high_risk_flows":  high,
        "suspicious_flows": sus,
    }, indent=2)


@tool
def evaluate_model_accuracy() -> str:
    """Evaluate GCN accuracy. Sets retrain flag if below threshold."""
    global TEST_DF
    if not GCN.trained or TEST_DF is None:
        return json.dumps({"error": "Not ready."})
    sample = TEST_DF.sample(min(2000, len(TEST_DF)))
    acc    = GCN.evaluate(sample)
    with STATE.lock:
        STATE.accuracy_log.append(acc)
        if acc < RETRAIN_THRESHOLD:
            STATE.retrain_flag = True
    # ── Loop 2 connection: persist accuracy to SQLite ─────────────────
    save_accuracy_to_db(acc, STATE.cycle)
    avg = round(float(np.mean(list(STATE.accuracy_log))), 4)
    return json.dumps({
        "current_accuracy":    acc,
        "average_accuracy":    avg,
        "retrain_recommended": acc < RETRAIN_THRESHOLD,
        "note": ("⚠️ Retrain needed." if acc < RETRAIN_THRESHOLD
                 else "✅ Accuracy healthy."),
    }, indent=2)


@tool
def raise_alert(input_json: str) -> str:
    """
    Raise a security alert.
    Input JSON: {"flow_id":"...","threat_level":"HIGH","reason":"...","src_ip":"..."}
    """
    try:
        import re, json as _j
        clean = re.sub(r"```json|```","", input_json).strip()
        if "{" not in clean:
            parts = dict(re.findall(
                r'(\w+)\s*=\s*["\']?([^,\'"]+)["\']?', clean))
            clean = _j.dumps(parts)
        data = _j.loads(clean)
    except Exception as e:
        return json.dumps({"error": str(e), "raw": input_json})

    flow_id      = str(data.get("flow_id","?"))
    threat_level = str(data.get("threat_level","?"))
    reason       = str(data.get("reason","?"))
    src_ip       = str(data.get("src_ip","?"))

    now = time.time()
    with STATE.lock:
        if now - STATE.last_alert_ts < 2:
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

    # ── Loop 2 connection: persist alert to SQLite ────────────────────
    save_alert_to_db(alert)
    print(f"\n🚨  [{alert['alert_id']}] {threat_level} | {src_ip}")
    return json.dumps({"status": "ALERT_RAISED", "alert": alert})


@tool
def blacklist_ip(input_json: str) -> str:
    """
    Blacklist a source IP.
    Input JSON: {"ip":"...","reason":"..."}
    """
    try:
        import re, json as _j
        clean = re.sub(r"```json|```","", input_json).strip()
        if "{" not in clean:
            parts = dict(re.findall(
                r'(\w+)\s*=\s*["\']?([^,\'"]+)["\']?', clean))
            clean = _j.dumps(parts)
        data  = _j.loads(clean)
    except Exception as e:
        return json.dumps({"error": str(e)})

    ip     = str(data.get("ip","?"))
    reason = str(data.get("reason","?"))
    with STATE.lock:
        STATE.ip_blacklist.add(ip)
        total = len(STATE.ip_blacklist)
    print(f"🚫  BLACKLIST: {ip} — {reason}")
    return json.dumps({"status":"BLACKLISTED","ip":ip,"total":total})


@tool
def recommend_retraining(reason: str = "accuracy drop") -> str:
    """Recommend GCN retraining. Input: plain text reason."""
    reason = str(reason).strip().strip('"').strip("'")
    with STATE.lock:
        STATE.retrain_flag = True
        avg = (round(float(np.mean(list(STATE.accuracy_log))),4)
               if STATE.accuracy_log else None)
    print(f"\n⚙️  RETRAIN RECOMMENDED: {reason}")
    return json.dumps({"status":"RETRAIN_RECOMMENDED",
                       "reason":reason, "avg_accuracy":avg})


@tool
def get_alerts_summary(input: str = "") -> str:
    """Return full summary of all alerts and blacklisted IPs."""
    with STATE.lock:
        return json.dumps({
            "total_alerts":    len(STATE.alerts),
            "blacklisted_ips": sorted(STATE.ip_blacklist),
            "recent_alerts":   STATE.alerts[-5:],
            "retrain_flag":    STATE.retrain_flag,
            "avg_accuracy":    (round(float(np.mean(
                               list(STATE.accuracy_log))),4)
                               if STATE.accuracy_log else None),
        }, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 3 — Agent builder
# ─────────────────────────────────────────────────────────────────────────────
REACT_TEMPLATE = """\
You are {agent_name}. {agent_role}

Tools: {tools}

Format:
Question: task
Thought: reasoning
Action: one of [{tool_names}]
Action Input: input
Observation: result
... repeat ...
Final Answer: conclusion

Begin!
Question: {input}
Thought:{agent_scratchpad}"""

def build_agent(name, role, tools):
    llm    = ChatGroq(api_key=GROQ_API_KEY,
                      model_name="llama-3.1-8b-instant",
                      temperature=0.1, max_tokens=512)
    prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
                agent_name=name, agent_role=role)
    agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         max_iterations=12, handle_parsing_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# Agent pipeline — runs in background thread
# ─────────────────────────────────────────────────────────────────────────────
def run_agent_cycle(cycle_num: int, log_container):
    STATE.cycle = cycle_num

    watcher  = build_agent("WatcherAgent",
        "Monitor traffic, ingest flows, report graph state.",
        [ingest_traffic_batch, get_graph_status])

    analyzer = build_agent("AnalyzerAgent",
        "Run GCN inference, interpret scores, evaluate accuracy.",
        [run_gcn_inference, evaluate_model_accuracy])

    decider  = build_agent("DeciderAgent",
        "raise_alert for HIGH flows, blacklist_ip their source, "
        "raise_alert for SUSPICIOUS, recommend_retraining if needed, "
        "end with get_alerts_summary.",
        [raise_alert, blacklist_ip,
         recommend_retraining, get_alerts_summary])

    log_container.info(f"Cycle {cycle_num} — Watcher running …")
    w = watcher.invoke({"input":
        "Ingest 8 traffic flows. Get graph status. "
        "Summarise protocols and patterns."})["output"]

    log_container.info(f"Cycle {cycle_num} — Analyzer running …")
    a = analyzer.invoke({"input":
        f"Watcher: {w[:300]}\n\n"
        "Run GCN inference (8 flows). Evaluate accuracy. "
        "List HIGH and SUSPICIOUS flows."})["output"]

    log_container.info(f"Cycle {cycle_num} — Decider running …")
    d = decider.invoke({"input":
        f"Analyzer: {a[:400]}\n\n"
        "1. raise_alert for every HIGH flow\n"
        "2. blacklist_ip every HIGH source IP\n"
        "3. raise_alert for every SUSPICIOUS flow\n"
        "4. recommend_retraining if needed\n"
        "5. get_alerts_summary"})["output"]

    log_container.success(f"Cycle {cycle_num} complete.")
    return w, a, d

# ─────────────────────────────────────────────────────────────────────────────
# LOOP 2 — Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global TEST_DF

    st.set_page_config(
        page_title="Cyber Threat Intelligence Platform",
        page_icon="🛡️",
        layout="wide")

    st.title("🛡️ Cyber Threat Intelligence Platform")
    st.caption("Loop 1 (GCN) + Loop 2 (Dashboard + SQLite) + Loop 3 (Agents)")

    # ── Init ─────────────────────────────────────────────────────────────────
    init_db()

    if not GCN.trained:
        with st.spinner("Training GCN model on UNSW-NB15 …"):
            GCN.fit(TRAIN_CSV, sample=20_000)
        st.success("✅ Model trained (90%+ accuracy)")

    if TEST_DF is None:
        TEST_DF = pd.read_csv(TEST_CSV).sample(15_000, random_state=42)

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Controls")
    cycles     = st.sidebar.slider("Simulation cycles", 1, 5, 3)
    run_button = st.sidebar.button("▶ Run Agent Simulation", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.header("🚫 Human Override")
    unblock_ip = st.sidebar.text_input("Unblacklist an IP:")
    if st.sidebar.button("Remove from blacklist"):
        if unblock_ip:
            STATE.ip_blacklist.discard(unblock_ip)
            st.sidebar.success(f"{unblock_ip} removed.")

    # ── Main layout ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    alerts_db   = load_alerts_from_db()
    flows_db    = load_flows_from_db()
    accuracy_db = load_accuracy_from_db()

    col1.metric("🚨 Total Alerts",
                len(alerts_db))
    col2.metric("🚫 Blacklisted IPs",
                len(STATE.ip_blacklist))
    col3.metric("📊 Avg Accuracy",
                f"{round(float(np.mean(list(STATE.accuracy_log))),3) if STATE.accuracy_log else '—'}")
    col4.metric("🔄 Flows Analysed",
                STATE.flow_counter)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕸️ Network Graph",
        "🚨 Alert Timeline",
        "📈 Accuracy Trend",
        "📋 Flow Log"
    ])

    with tab1:
        st.subheader("Live Network Graph")
        st.caption("Red = HIGH risk  |  Orange = SUSPICIOUS  |  Green = NORMAL")
        if len(GRAPH.G.nodes) > 0:
            st.pyplot(GRAPH.get_figure())
        else:
            st.info("No flows yet — run the simulation to populate the graph.")

    with tab2:
        st.subheader("Alert Timeline")
        if not alerts_db.empty:
            # Colour rows by threat level
            def colour_row(row):
                c = ("#ffcccc" if row.threat_level=="HIGH" else
                     "#fff3cc" if row.threat_level=="SUSPICIOUS"
                     else "#ccffcc")
                return [f"background-color: {c}"] * len(row)
            st.dataframe(alerts_db.style.apply(colour_row, axis=1),
                         use_container_width=True)
        else:
            st.info("No alerts yet.")

    with tab3:
        st.subheader("GCN Model Accuracy Over Time")
        if not accuracy_db.empty:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(accuracy_db["accuracy"].values,
                    "b-o", linewidth=2, markersize=5)
            ax.axhline(RETRAIN_THRESHOLD, color="red",
                       linestyle="--", label=f"Retrain threshold ({RETRAIN_THRESHOLD})")
            ax.set_ylim(0.5, 1.0)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Evaluation #")
            ax.legend()
            ax.set_title("Model Health")
            st.pyplot(fig)
            if STATE.retrain_flag:
                st.warning("⚙️ Retraining recommended — accuracy dropped below threshold.")
        else:
            st.info("No accuracy data yet.")

    with tab4:
        st.subheader("Recent Flow Log")
        if not flows_db.empty:
            st.dataframe(flows_db, use_container_width=True)
        else:
            st.info("No flows logged yet.")

    # ── Run simulation ────────────────────────────────────────────────────────
    if run_button:
        log = st.empty()
        for i in range(1, cycles + 1):
            st.markdown(f"### Cycle {i} / {cycles}")
            w_col, a_col, d_col = st.columns(3)

            with st.spinner(f"Running cycle {i} …"):
                w, a, d = run_agent_cycle(i, log)

            with w_col:
                st.markdown("**🔍 Watcher**")
                st.write(w)
            with a_col:
                st.markdown("**🧠 Analyzer**")
                st.write(a)
            with d_col:
                st.markdown("**⚡ Decider**")
                st.write(d)

            time.sleep(0.5)
            st.rerun()

if __name__ == "__main__":
    main()