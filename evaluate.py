from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 2. RETRIEVAL METRICS
# ----------------------------------------

def hit_rate_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Checks whether at least one relevant document
    appears in the top K retrieved documents.
    """
    top_k = retrieved_ids[:k]

    return int(any(doc_id in relevant_ids for doc_id in top_k))


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Finds the rank of the first relevant document.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


# Sample evaluation queries with expected rulebook phrases
EVAL_DATASET = [
    {
        "question": "What is the policy on equipment?",
        "keywords": ["Acceptable use of company assets", "Company laptops", "Keep equipment secure"],
    },
    {
        "question": "What are the working hours and attendance guidelines?",
        "keywords": ["Working hours and attendance", "Standard working hours", "record attendance"],
    },
    {
        "question": "What are the rules for travel and expenses?",
        "keywords": ["Expenses and travel", "business travel", "itemised receipts"],
    },
    {
        "question": "What is the policy on information security and privacy?",
        "keywords": ["Information security and privacy", "Protect passwords", "multi-factor authentication"],
    },
    {
        "question": "How should leave and time off be requested?",
        "keywords": ["Leave and time off", "planned leave", "Emergency or sick leave"],
    },
]


def find_relevant_ids(all_ids, all_docs, keywords):
    """Find chunk IDs whose text contains any of the expected keywords."""
    relevant = set()
    for doc_id, text in zip(all_ids, all_docs):
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in keywords):
            relevant.add(doc_id)
    return relevant


def main():
    k = 5
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    stored = vector_db.get(include=["documents", "metadatas"])
    all_ids = stored.get("ids", [])
    all_docs = stored.get("documents", [])

    if not all_ids:
        print("No documents found in ./chroma_db. Run `python3 ingest.py` first.")
        return

    hit_rates = []
    reciprocal_ranks = []

    for item in EVAL_DATASET:
        question = item["question"]
        relevant_ids = find_relevant_ids(all_ids, all_docs, item["keywords"])

        results = vector_db.similarity_search(question, k=k)
        retrieved_ids = [
            doc.metadata.get("chunk_id") or doc.id
            for doc in results
        ]

        hr = hit_rate_at_k(retrieved_ids, relevant_ids, k=k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        hit_rates.append(hr)
        reciprocal_ranks.append(rr)

        print(f"Question: {question}")
        print(f"  Relevant IDs : {sorted(relevant_ids)}")
        print(f"  Retrieved IDs: {retrieved_ids}")
        print(f"  Hit Rate@{k}  : {hr}")
        print(f"  MRR (RR)     : {rr:.4f}\n")

    avg_hit_rate = sum(hit_rates) / len(hit_rates)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("========================================")
    print(f"Overall Hit Rate@{k}: {avg_hit_rate:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")


if __name__ == "__main__":
    main()
