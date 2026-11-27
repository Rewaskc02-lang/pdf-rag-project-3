# PDF RAG Project

A simple and lightweight Retrieval-Augmented Generation (RAG) pipeline for extracting, cleaning, chunking, embedding, and querying documents from PDF files.

## Project Structure

```text
├── northstar_company_rulebook.pdf  # Source PDF document
├── clean_pdf.py                   # Cleans headers, page numbers, and whitespace
├── cleaned_rulebook.txt           # Cleaned text output extracted from the PDF
├── ingest.py                      # Chunks text and creates ChromaDB vector store
├── query.py                       # Queries the vector database and retrieves answers
├── evaluate.py                    # Evaluates retrieval quality (Hit Rate@K and MRR)
├── index.txt                      # Sample questions for testing
└── README.md                      # Project documentation
```

## Prerequisites & Installation

Install the required Python packages:

```bash
pip install pypdf langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers
```

## How to Run

### Step 1: Clean the PDF
Extracts text from the PDF, removes repeating headers and page numbers, and standardizes spacing:

```bash
python3 clean_pdf.py
```
*Output: `cleaned_rulebook.txt`*

### Step 2: Ingest into Vector Database
Splits the cleaned text into chunks, generates vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` locally (free), and stores them in `./chroma_db`:

```bash
python3 ingest.py
```

### Step 3: Query the Knowledge Base
Search the Chroma vector database for answers either interactively or by passing a question as a command-line argument:

```bash
python3 query.py
```

Or pass your question directly:

```bash
python3 query.py "What is the policy on equipment?"
```

Sample questions you can try:
- *What is the policy on equipment?*
- *What are the working hours and attendance guidelines?*
- *What are the rules for travel and expenses?*

### Step 4: Evaluate Retrieval Metrics
Measure retrieval performance using **Hit Rate@K** and **Mean Reciprocal Rank (MRR)** against sample rulebook queries:

```bash
python3 evaluate.py
```

## Tech Stack
- **PDF Processing**: `pypdf`
- **RAG Framework**: `langchain`
- **Vector Database**: `ChromaDB`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (local Hugging Face embeddings)
