"""
Mock LLM — deterministic ReAct steps for demo without any API key.
Auto-detected and used by run_simulation.py when GROQ_API_KEY is not set.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Optional, List, Any

SEQUENCES = {
    "watcher": [
        (" Ingesting traffic.\nAction: ingest_traffic_batch\nAction Input: 8"),
        (" Checking graph.\nAction: get_graph_status\nAction Input: "),
        (" Done.\nFinal Answer: Ingested 8 flows (TCP/UDP/ICMP). Graph updated. "
         "Observed Normal + potential DoS/Fuzzers. Handing off to Analyzer."),
    ],
    "analyzer": [
        (" Running GCN inference.\nAction: run_gcn_inference\nAction Input: 8"),
        (" Checking accuracy.\nAction: evaluate_model_accuracy\nAction Input: "),
        (" Done.\nFinal Answer: 2 HIGH flows (risk>0.80), 1 SUSPICIOUS (risk~0.67), 5 NORMAL. "
         "Accuracy 0.89 — healthy, no retrain needed. "
         "HIGH: flow_00001 src=10.3.147.22 score=0.92 (DoS), "
         "flow_00003 src=10.1.55.88 score=0.84 (Exploit). "
         "SUSPICIOUS: flow_00004 src=10.1.88.203 score=0.67. Passing to Decider."),
    ],
    "decider": [
        (" Raising alert for HIGH flow.\nAction: raise_alert\n"
         "Action Input: flow_id=flow_00001 threat_level=HIGH "
         "reason=GCN risk 0.92 DoS pattern src_ip=10.3.147.22"),
        (" Blacklisting HIGH source IP.\nAction: blacklist_ip\n"
         "Action Input: ip=10.3.147.22 reason=HIGH DoS risk 0.92"),
        (" Raising alert for second HIGH.\nAction: raise_alert\n"
         "Action Input: flow_id=flow_00003 threat_level=HIGH "
         "reason=GCN risk 0.84 Exploit pattern src_ip=10.1.55.88"),
        (" Blacklisting second IP.\nAction: blacklist_ip\n"
         "Action Input: ip=10.1.55.88 reason=HIGH Exploit risk 0.84"),
        (" Raising SUSPICIOUS alert.\nAction: raise_alert\n"
         "Action Input: flow_id=flow_00004 threat_level=SUSPICIOUS "
         "reason=GCN risk 0.67 anomalous connection src_ip=10.1.88.203"),
        (" Getting final summary.\nAction: get_alerts_summary\nAction Input: "),
        (" All done.\nFinal Answer: Raised 3 alerts (2 HIGH, 1 SUSPICIOUS). "
         "Blacklisted: 10.3.147.22 (DoS), 10.1.55.88 (Exploit). "
         "Accuracy 0.89 — no retrain needed. System autonomously secure."),
    ],
}


class MockChatLLM(BaseChatModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_counters", {})

    @property
    def _llm_type(self) -> str:
        return "mock_react_llm"

    def _detect_agent(self, messages) -> str:
        txt = " ".join(str(m.content) for m in messages).lower()
        if "watcheragent" in txt or ("ingest" in txt and "monitor" in txt):
            return "watcher"
        if "analyzeragent" in txt or "gcn inference" in txt:
            return "analyzer"
        return "decider"

    def _generate(self, messages: List, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        agent    = self._detect_agent(messages)
        seq      = SEQUENCES[agent]
        counters = object.__getattribute__(self, "_counters")
        idx      = counters.get(agent, 0)
        step     = seq[idx % len(seq)]          # cycle so multi-cycle sim works
        counters[agent] = idx + 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=step))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)