from cross_encoder import rerank_documents
from vector_database import collection

# Replace these examples with questions and distinctive relevant phrases
# that actually appear in your own PDF.
test_dataset = [
    {
        "question": "What is the policy on equipment?",
        "relevant_text": "Company laptops, software, accounts, networks"
    },
    {
        "question": "What are the working hours and attendance guidelines?",
        "relevant_text": "Standard working hours are set by the relevant team"
    },
    {
        "question": "What are the rules for travel and expenses?",
        "relevant_text": "Obtain approval before booking business travel"
    },
    {
        "question": "What is the policy on information security and privacy?",
        "relevant_text": "Protect passwords, authentication codes, devices"
    },
    {
        "question": "How should leave and time off be requested?",
        "relevant_text": "Submit planned leave through the designated HR system"
    },
]


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


def evaluate_retrieval(dataset, k=5):
    if not dataset:
        print("Test dataset is empty.")
        return 0.0, 0.0

    stored = collection.get(include=["documents"])
    all_ids = stored.get("ids", [])
    all_docs = stored.get("documents", [])

    if not all_ids:
        print("Vector database is empty. Run vector_database.py first.")
        return 0.0, 0.0

    hit_rates = []
    reciprocal_ranks = []

    for item in dataset:
        question = item["question"]
        relevant_phrase = item["relevant_text"].casefold()

        relevant_ids = {
            doc_id
            for doc_id, text in zip(all_ids, all_docs)
            if relevant_phrase in text.casefold()
        }

        # Evaluate the final reranked top-K results.
        results = rerank_documents(question, top_k=k, fetch_k=10)
        retrieved_ids = [result["id"] for result in results]

        hr = hit_rate_at_k(retrieved_ids, relevant_ids, k=k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        hit_rates.append(hr)
        reciprocal_ranks.append(rr)

        print(f"Question: {question}")
        print(f"  Relevant IDs : {sorted(relevant_ids)}")
        print(f"  Retrieved IDs: {retrieved_ids}")
        print(f"  Hit Rate@{k}  : {hr}")
        print(f"  Reciprocal Rank: {rr:.4f}\n")

    hit_rate_score = sum(hit_rates) / len(dataset)
    mrr_score = sum(reciprocal_ranks) / len(dataset)

    print("========== RETRIEVAL EVALUATION ==========")
    print(f"Total questions: {len(dataset)}")
    print(f"Successful hits: {sum(hit_rates)}")
    print(f"Hit Rate@{k}: {hit_rate_score:.2f} ({hit_rate_score * 100:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    return hit_rate_score, mrr_score


if __name__ == "__main__":
    evaluate_retrieval(test_dataset, k=5)
