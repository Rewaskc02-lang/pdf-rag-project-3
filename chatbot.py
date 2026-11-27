import os
from dotenv import load_dotenv
from google import genai

from cross_encoder import rerank_documents

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Copy .env.example to .env and add your API key."
    )

# Change the model name here if your Google AI Studio account does not offer it.
GEMINI_MODEL = "gemini-3.8-flash"

# Cross-Encoder scores are not probabilities. This is only a starter threshold;
# test and tune it using questions from your own PDF.
MIN_RERANK_SCORE = 0.0

client = genai.Client(api_key=API_KEY)


def ask_question_with_sources(question):
    """Answer a question using reranked chunks and return (answer, documents)."""
    documents = rerank_documents(question, top_k=5, fetch_k=10)

    if not documents:
        return "I don't know. The answer is not in the document.", []

    best_score = documents[0]["rerank_score"]
    if best_score < MIN_RERANK_SCORE:
        return "I don't know. The answer is not in the document.", documents

    context = "\n\n".join(
        f"[Page {doc['page']}] {doc['text']}" for doc in documents
    )

    prompt = f"""
You are a helpful company rulebook assistant.
Answer the question using ONLY the context provided below.
If the context does not contain the answer, reply exactly:
I don't know. The answer is not in the document.
Do not use outside knowledge or invent facts.
Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    answer = getattr(response, "text", None)
    return (answer.strip() if answer else "No text response was returned."), documents


def ask_question(question):
    answer, _ = ask_question_with_sources(question)
    return answer


if __name__ == "__main__":
    print("\n===== PDF RAG CHATBOT =====")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Chatbot closed.")
            break
        if not question:
            continue

        try:
            print("\nBot:", ask_question(question))
        except Exception as error:
            print(f"An error occurred: {error}")
        print("\n" + "-" * 55)
