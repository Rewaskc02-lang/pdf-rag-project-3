import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = input("Enter your question: ")

if not question.strip():
    question = "What are the employee guidelines?"

results = vector_db.similarity_search(question, k=3)

for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print()
