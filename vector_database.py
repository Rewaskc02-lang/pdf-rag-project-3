from pathlib import Path
import json
import chromadb

from embeddings import create_embeddings

CHUNKS_PATH = Path("data/chunks.json")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def store_vectors():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Run chunk_document.py first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("The chunks file is empty.")

    # Rebuild this demo collection from the current chunks.
    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    texts = [chunk["text"] for chunk in chunks]
    vectors = create_embeddings(texts)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=vectors.tolist(),
        metadatas=[{"page": int(chunk["page"])} for chunk in chunks]
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


def retrieve_documents(question, top_k=5):
    count = collection.count()
    if count == 0:
        return []

    query_vector = create_embeddings([question])
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i, chunk_id in enumerate(results["ids"][0]):
        retrieved.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })
    return retrieved


if __name__ == "__main__":
    store_vectors()
    sample_results = retrieve_documents(
        "What is the policy on equipment?",
        top_k=5
    )
    for item in sample_results:
        print("\nPage:", item["page"])
        print("Distance:", round(item["distance"], 4))
        print("Text:", item["text"][:300])
