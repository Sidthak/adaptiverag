# app.py
# Streamlit chat UI for AdaptiveRAG.
# Run with: streamlit run app.py
 
import streamlit as st
import uuid
from agent import run_agent
from retriever import has_documents
 
st.set_page_config(
    page_title="AdaptiveRAG",
    page_icon="🧠",
    layout="centered",
)
 
with st.sidebar:
    st.title("🧠 AdaptiveRAG")
    st.caption("Smart assistant that knows where to look")
    st.divider()
    st.markdown("""
**How it works:**
1. You ask any question
2. It decides: web, docs, or both
3. It grades the results (CRAG)
4. If unsure, it **pauses** and asks YOU (HITL)
5. You answer → it resumes and answers
""")
    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"]
    )
    if uploaded_files:
        import os
        os.makedirs("./docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(f"./docs/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) saved!")
        st.info("Now run python ingest.py in terminal to index them.")
 
    st.divider()
    doc_status = "✅ Documents indexed" if has_documents() else "⚠️ No documents yet"
    st.caption(doc_status)
 
    if st.session_state.get("paused_thread"):
        st.warning("⏸️ Graph paused — waiting for your input!")
 
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.paused_thread = None
        st.rerun()
 
st.title("🧠 AdaptiveRAG")
st.caption("Ask anything — I'll figure out the best way to answer")
 
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
 
if "paused_thread" not in st.session_state:
    st.session_state.paused_thread = None
 
# Show paused status banner
if st.session_state.paused_thread:
    st.info("⏸️ **Graph is paused** — type your answer below to resume!")
 
# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("route"):
            st.caption(f"🔀 Routed to: **{msg['route']}**")
        if msg.get("clarification"):
            st.warning(f"❓ {msg['clarification']}")
 
# Chat input
if query := st.chat_input("Ask me anything..." if not st.session_state.paused_thread else "Type your answer to resume..."):
 
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
 
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and routing..." if not st.session_state.paused_thread else "▶️ Resuming from where we left off..."):
 
            # Check if we are resuming a paused graph
            if st.session_state.paused_thread:
                result = run_agent(
                    query,
                    st.session_state.paused_thread,
                    human_input=query
                )
                st.session_state.paused_thread = None
            else:
                result = run_agent(query, st.session_state.thread_id)
 
        # Handle paused graph (HITL triggered)
        if result.get("needs_clarification") or result.get("graph_paused"):
            clarification = result.get("clarification_question", "")
            st.warning(f"⏸️ **Graph paused!** I need more information:\n\n❓ {clarification}")
            st.session_state.paused_thread = st.session_state.thread_id
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⏸️ Graph paused — I need more info: {clarification}",
                "route": result.get("route", ""),
                "clarification": clarification,
            })
 
        # Handle normal answer
        else:
            answer = result.get("answer", "")
            route = result.get("route", "")
            st.write(answer)
            st.caption(f"🔀 Routed to: **{route}**")
 
            if result.get("web_results"):
                with st.expander("🌐 Web sources"):
                    web_results = list(result["web_results"])
                    for r in web_results[:3]:
                        if isinstance(r, dict):
                            st.write(f"- {r.get('url', r.get('content', '')[:100])}")
                        else:
                            st.write(f"- {str(r)[:100]}")
 
            if result.get("doc_results"):
                with st.expander("📄 Document sources"):
                    doc_results = list(result["doc_results"])
                    for r in doc_results[:3]:
                        st.write(f"- {r.get('source', 'unknown')}")
 
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "route": route,
            })