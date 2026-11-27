from sentence_transformers import CrossEncoder

from vector_database import retrieve_documents

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_NAME)


def rerank_documents(question, top_k=5, fetch_k=10):
    # Retrieve candidates with the embedding model first.
    candidates = retrieve_documents(question, top_k=fetch_k)
    if not candidates:
        return []

    pairs = [[question, item["text"]] for item in candidates]
    scores = reranker.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    results = rerank_documents(
        "What is the policy on equipment?",
        top_k=5,
        fetch_k=10
    )
    for item in results:
        print(
            f"Page {item['page']} | "
            f"Rerank score: {item['rerank_score']:.4f}\n"
            f"{item['text'][:300]}\n"
        )
