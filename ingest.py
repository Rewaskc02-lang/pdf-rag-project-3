from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = TextLoader("cleaned_rulebook.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
for chunk_id, chunk in zip(chunk_ids, chunks):
    chunk.metadata["chunk_id"] = chunk_id

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=chunk_ids,
    persist_directory="./chroma_db"
)

print("Done! Database is ready in ./chroma_db folder.")
