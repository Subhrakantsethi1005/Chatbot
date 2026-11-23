"""Read-only monitoring dashboard (needs a viewer, operator or admin token)."""

import streamlit as st

from kb_pipeline.config import load_config
from kb_pipeline.control import ControlPlane
from kb_pipeline.errors import AccessDenied

st.set_page_config(page_title="KB Monitoring", page_icon="📈")
st.title("Knowledge-base monitoring")

token = st.text_input("Access token", type="password", help="Create one with: python kbctl.py user-add --role viewer")
window = st.selectbox("Time window", [15, 60, 240, 1440], index=1, format_func=lambda m: f"last {m} minutes")

if token:
    plane = ControlPlane(load_config(), client_id="web-monitoring")
    try:
        status = plane.status(token)
        summary = plane.metrics(token, window_minutes=window)
        versions = plane.versions(token)
        alerts = plane.alerts(token, limit=15)
        quarantine = plane.quarantine(token)
    except AccessDenied as exc:
        st.error(f"Access denied: {exc}")
        st.stop()

    active = status.get("active") or {}
    chat = summary["chat"]
    cols = st.columns(4)
    cols[0].metric("Active version", active.get("version") or "none")
    cols[1].metric("Requests", chat["requests"])
    cols[2].metric("p95 latency (ms)", chat["latency_ms"]["p95"] or 0)
    cols[3].metric("Avg confidence", f"{(chat['avg_confidence'] or 0):.0%}")
    cols = st.columns(4)
    cols[0].metric("Error rate", f"{chat['error_rate']:.1%}")
    cols[1].metric("Escalation rate", f"{chat['escalation_rate']:.1%}")
    cols[2].metric("Blocked (injection)", summary["security"]["injection_blocked"])
    cols[3].metric("Denied access", summary["security"]["auth_denied"])

    st.subheader("Pipeline")
    st.write({
        "maintenance window": status["maintenance_window"],
        "next scheduled update": status["schedule"]["next_run"],
        "pending activation": status["pending"],
        "health watch": status["health_watch"],
        "retry": status["retry"],
        "last update": status["last_update"],
    })
    st.write({"updates": summary["updates"], "rollbacks": summary["rollbacks"],
              "health checks": summary["health_checks"]})

    st.subheader("Versions")
    st.dataframe(versions)
    st.subheader("Recent alerts")
    st.dataframe(alerts or [{"message": "no alerts"}])
    st.subheader("Quarantined files")
    st.dataframe(quarantine or [{"path": "none"}])
