import streamlit as st

from chatbot import ask_question_with_sources

st.set_page_config(
    page_title="Northstar Rulebook RAG Chatbot",
    page_icon="📘",
    layout="centered",
)

st.title("Northstar Company Rulebook Chatbot")
st.caption("Ask questions about the Northstar Employee Rulebook (ChromaDB + Cross-Encoder + Gemini)")

SAMPLE_QUESTIONS = [
    "What is the policy on equipment?",
    "What are the working hours and attendance guidelines?",
    "What are the rules for travel and expenses?",
    "What is the policy on information security and privacy?",
    "How should leave and time off be requested?",
]

with st.sidebar:
    st.subheader("Sample Questions")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample, use_container_width=True):
            st.session_state["pending_question"] = sample

    if st.button("Clear Chat", type="secondary", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved Sources"):
                for doc in msg["sources"]:
                    st.markdown(
                        f"**Page {doc['page']}** (Rerank score: `{doc['rerank_score']:.4f}`)\n\n"
                        f"> {doc['text']}"
                    )

prompt = st.chat_input("Ask a question about the rulebook...")
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching rulebook and generating answer..."):
            try:
                answer, sources = ask_question_with_sources(prompt)
            except Exception as error:
                answer = f"Error: {error}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Retrieved Sources"):
                for doc in sources:
                    st.markdown(
                        f"**Page {doc['page']}** (Rerank score: `{doc['rerank_score']:.4f}`)\n\n"
                        f"> {doc['text']}"
                    )

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
