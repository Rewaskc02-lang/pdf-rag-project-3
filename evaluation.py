from cross_encoder import rerank_documents
from vector_database import collection

# Replace these examples with questions and distinctive relevant phrases
# that actually appear in your own PDF.
test_dataset = [
    {
        "question": "What is the policy on equipment?",
        "relevant_text": "Keep equipment secure"
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


def evaluate_retrieval(dataset, k_values=(1, 2, 3, 5)):
    if not dataset:
        print("Test dataset is empty.")
        return None

    stored = collection.get(include=["documents"])
    all_ids = stored.get("ids", [])
    all_docs = stored.get("documents", [])

    if not all_ids:
        print("Vector database is empty. Run vector_database.py first.")
        return None

    max_k = max(k_values)
    hit_totals = {k: 0 for k in k_values}
    reciprocal_ranks = []
    per_query_results = []

    for item in dataset:
        question = item["question"]
        relevant_phrase = item["relevant_text"].casefold()

        relevant_ids = {
            doc_id
            for doc_id, text in zip(all_ids, all_docs)
            if relevant_phrase in text.casefold()
        }

        # Evaluate the final reranked top-K results.
        results = rerank_documents(question, top_k=max_k, fetch_k=10)
        retrieved_ids = [result["id"] for result in results]

        hits_for_query = {}
        for k in k_values:
            hr = hit_rate_at_k(retrieved_ids, relevant_ids, k=k)
            hits_for_query[k] = hr
            hit_totals[k] += hr

        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        per_query_results.append({
            "question": question,
            "relevant_ids": sorted(relevant_ids),
            "retrieved_ids": retrieved_ids,
            "hits": hits_for_query,
            "reciprocal_rank": rr,
        })

        print(f"Question: {question}")
        print(f"  Relevant IDs   : {sorted(relevant_ids)}")
        print(f"  Retrieved IDs  : {retrieved_ids}")
        for k in k_values:
            print(f"  Hit@{k}          : {hits_for_query[k]}")
        print(f"  Reciprocal Rank: {rr:.4f}\n")

    total = len(dataset)
    hit_rates = {k: hit_totals[k] / total for k in k_values}
    mrr_score = sum(reciprocal_ranks) / total

    print("========== RETRIEVAL EVALUATION ==========")
    print(f"Total questions: {total}")
    for k in k_values:
        print(f"Hit@{k}: {hit_rates[k]:.2f} ({hit_rates[k] * 100:.2f}%) [{hit_totals[k]}/{total}]")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

    return {
        "total": total,
        "hit_rates": hit_rates,
        "mrr": mrr_score,
        "per_query": per_query_results,
    }


if __name__ == "__main__":
    evaluate_retrieval(test_dataset, k_values=(1, 2, 3, 5))
