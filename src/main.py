"""Customer-service chatbot UI.

Answers come from the *active* knowledge-base version managed by the
pipeline (``python kbctl.py update``); there is no button here that rebuilds
the knowledge base, so a bad update can never be pushed live from the UI.
Run with:  streamlit run src/main.py
"""

import uuid

import streamlit as st

from kb_pipeline.config import load_config
from kb_pipeline.errors import AccessDenied
from kb_pipeline.service import ChatService

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")


@st.cache_resource
def get_service() -> ChatService:
    return ChatService(load_config())


service = get_service()
client_id = st.session_state.setdefault("client_id", f"web-{uuid.uuid4().hex[:12]}")

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
question = st.text_input("Question: ")

if question:
    try:
        response = service.ask(question, client_id=client_id)
    except AccessDenied as exc:
        st.error(f"Access denied: {exc}")
    else:
        st.header("Answer")
        st.write(response.answer)
        if response.blocked:
            st.warning("This request was blocked by the safety filter.")
        elif response.escalated:
            st.info(f"Passed to our support team (ticket {response.ticket_id}).")
        if response.version:
            st.caption(f"Confidence {response.confidence:.0%} · knowledge base {response.version}"
                       + (" · degraded mode" if response.degraded else ""))
