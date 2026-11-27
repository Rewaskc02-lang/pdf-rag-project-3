from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded once and reused by the importing files.
embedding_model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    if not texts:
        return []
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vectors


if __name__ == "__main__":
    sample_vectors = create_embeddings(["What is the policy on equipment?"])
    print("Embedding shape:", sample_vectors.shape)
    print("First 10 values:", sample_vectors[0][:10])
