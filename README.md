# Basic PDF RAG Chatbot (Python + ChromaDB + Cross-Encoder)

A beginner-friendly, multi-file PDF RAG project configured for the Northstar Company Rulebook (`documents/northstar_company_rulebook.pdf`). It uses:
- PyPDF for PDF text extraction
- fixed-character chunking with overlap
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- a Cross-Encoder for reranking
- Gemini for grounded chatbot answers
- Hit Rate@K and Mean Reciprocal Rank (MRR) for retrieval evaluation

## Project Structure

```text
├── documents/
│   └── northstar_company_rulebook.pdf
├── load_document.py
├── clean_document.py
├── chunk_document.py
├── embeddings.py
├── vector_database.py
├── cross_encoder.py
├── evaluation.py
├── chatbot.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 1. Setup

Create and activate a virtual environment.

### macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Configure API key

1. Your PDF is located at `documents/northstar_company_rulebook.pdf`.
2. Copy `.env.example` to `.env`.
3. Add your Gemini API key to `.env`.

Get a key from https://aistudio.google.com/apikey

## 3. Run in order

Run these commands from this project folder:

```bash
python3 load_document.py
python3 clean_document.py
python3 chunk_document.py
python3 embeddings.py
python3 vector_database.py
python3 cross_encoder.py
python3 chatbot.py
```

Or launch the web UI with Streamlit:
```bash
streamlit run app.py
```

In a separate run, evaluate retrieval:
```bash
python3 evaluation.py
```

Run the ingestion steps again whenever you replace or update the PDF.

## 4. Configure evaluation questions

Edit `test_dataset` in `evaluation.py`. For each question, set `relevant_text` to a distinctive phrase that appears in a genuinely relevant PDF chunk. `evaluation.py` computes **Hit Rate@K** (`hit_rate_at_k`) and **Mean Reciprocal Rank** (`reciprocal_rank`).

## Notes

- First run downloads embedding and reranker models and may take several minutes.
- Gemini calls require internet access and a valid API key.
- The chatbot's refusal check is a simple relevance-score threshold, not a guarantee of perfect grounding. Adjust `MIN_RERANK_SCORE` after testing on your own documents.
- Cross-Encoder reranking can only reorder the candidates returned by vector search; it cannot find chunks that initial retrieval missed.
