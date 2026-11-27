import streamlit as st

from chatbot import ask_question_with_sources
from evaluation import evaluate_retrieval, test_dataset

st.set_page_config(
    page_title="Northstar Rulebook RAG Chatbot",
    page_icon="📘",
    layout="centered",
)

st.title("Northstar Company Rulebook Chatbot")
st.caption("ChromaDB + Cross-Encoder Reranking + Gemini + Hit@K / MRR Evaluation")

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

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation (Hit@K & MRR)"])

with tab_chat:
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

with tab_eval:
    st.subheader("Retrieval Evaluation Metrics")
    if st.button("Run Evaluation (Hit@1, Hit@2, Hit@3, Hit@5 & MRR)", type="primary"):
        with st.spinner("Running retrieval evaluation..."):
            st.session_state["eval_summary"] = evaluate_retrieval(
                test_dataset, k_values=(1, 2, 3, 5)
            )

    if st.session_state.get("eval_summary"):
        summary = st.session_state["eval_summary"]
        cols = st.columns(5)
        for idx, k in enumerate((1, 2, 3, 5)):
            cols[idx].metric(f"Hit@{k}", f"{summary['hit_rates'][k] * 100:.0f}%")
        cols[4].metric("MRR", f"{summary['mrr']:.4f}")

        st.divider()
        for row in summary["per_query"]:
            st.markdown(
                f"**{row['question']}**\n"
                + " | ".join(f"`Hit@{k}: {row['hits'][k]}`" for k in (1, 2, 3, 5))
                + f" | `RR: {row['reciprocal_rank']:.2f}`"
            )
