from pathlib import Path
import json

CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 100    # characters
INPUT_PATH = Path("data/cleaned.json")
OUTPUT_PATH = Path("data/chunks.json")


def create_chunks():
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Run clean_document.py first.")

    pages = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    chunks = []
    chunk_number = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page in pages:
        text = page["text"]
        for start in range(0, len(text), step):
            chunk_text = text[start:start + CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({
                    "id": f"chunk_{chunk_number}",
                    "page": page["page"],
                    "text": chunk_text
                })
                chunk_number += 1

    if not chunks:
        raise ValueError("No chunks were created. Check that your PDF contains text.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Created {len(chunks)} chunks. Saved to {OUTPUT_PATH}.")
    return chunks


if __name__ == "__main__":
    create_chunks()
