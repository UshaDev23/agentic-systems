from google import genai
import chromadb
from pypdf import PdfReader
import random
from pathlib import Path
from typing import List, Dict, Any
import re

# 1. Initialize GenAI client
genai_client = genai.Client()

# 2. Define Embedding model and LLM model
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.0-flash"


# Helper Functions

def load_pdf_file(file_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(file_path))
    documents = []
    for pg_number, pg in enumerate(reader.pages, start=1):
        pg_text = pg.extract_text()
        cleaned_text = clean_text(pg_text)
        if cleaned_text:
            documents.append(make_document(
                text = cleaned_text,
                metadata = {
                    "source": str(file_path),
                    "source_type":"pdf",
                    "page_number": pg_number,
                    "policy_type": infer_policy_type(file_path.name)
                }
            ))
    return documents

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_document(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": text,
        "metadata": metadata
    }

# Chunking - Fized Size Chunking with overlapping
def chunk_text(text: str, chunk_size: int = 120, overlap_words: int = 30):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start: end]
        chunk  = " ".join(chunk_words)
        chunks.append(chunk)

        if(end >= len(words)):
            break

        start = end - overlap_words
    return chunks

# 3. Infer policy type based on source name
def infer_policy_type(source_name: str) -> str:
    file_name = source_name.lower()
    if "hostel" in file_name:
        return "hostel"
    if "library" in file_name:
        return "library"
    if "refund" in file_name:
        return "refund"
    return "general"

# 4. Load all files from policy_documents folder
def load_docs_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    all_documents = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            all_documents.extend(load_pdf_file(file_path))
    
    return all_documents

# 5. Chunking documents into smaller pieces

def create_chunk(documents: List[Dict[str, Any]], 
                chunk_size_words: int = 120,
                overlap_words: int = 30) -> List[Dict[str, Any]]:
    all_chunks = []
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        text_chunks = chunk_text(text, chunk_size=chunk_size_words, overlap_words=overlap_words)
        for idx, chunk in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["chunk_size_words"] = chunk_size_words
            chunk_metadata["overlap_words"] = overlap_words

            source = chunk_metadata["source"]
            # unique_string = f"{source}_{idx}_{chunk}"
            chunk_id = f"{source}_{idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata
            })
    return all_chunks

# 6. Create Embeddings
def create_embeddings(text: List[str]) -> List[List[float]]:
    response = genai_client.models.embed_content(
        model = EMBEDDING_MODEL,
        contents = text
    )
    embeddings = [item.values for item in response.embeddings]
    return embeddings

# 7. Setup Vector database
def setup_vector_database():
    chroma_client = chromadb.PersistentClient(path = "./chroma_policy_db")
    collection = chroma_client.get_or_create_collection(
        name = "campus_policy_collection",
        embedding_function = None
    )
    return collection


# 8. Index the documents to vector db
def index_chunks(collection, chunks: List[Dict[str, Any]], batch_size: int = 50):
    if not chunks:
        print("No chunks to index")
        return
    
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        ids = [chunk["id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]

        embeddings = create_embeddings(texts)

        collection.upsert(
            ids = ids,
            documents = texts,
            metadatas = metadatas,
            embeddings = embeddings
        )

        print(f"Indexed batch {start // batch_size + 1}: {len(batch)} chunks")

# 9. Build knowledge base
def build_knowledge_base(
    collection,
    folder_path: str,
    chunk_size_words: int = 120,
    overlap_words: int = 30
) -> None:
    print("\nLoading documents from folder...")
    documents = load_docs_from_folder(folder_path)
    print(f"Total documents loaded: {len(documents)}")

    print("\nCreating chunks from documents...")
    chunks = create_chunk(documents, chunk_size_words, overlap_words)
    print(f"Total chunks created: {len(chunks)}")

    print("\nIndexing chunks to vector database...")
    index_chunks(collection, chunks)
    print("Knowledge base build complete.")

# 10. Retrieve relevant documents based on query
def retrieve_relevant_chunks(collection, query: str, top_k: int = 4, policy_type: str = None) -> List[Dict[str, Any]]:
    query_embedding = create_embeddings([query])[0]

    where_filter = None
    if policy_type:
        where_filter = {
            "policy_type": policy_type
        }
    
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k,
        where = where_filter,
        include = ["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved_chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })

    return retrieved_chunks

# 11. Building prompt for the LLm to generate the final answer
def build_grounded_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_parts =[]
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        source = metadata.get("source", "unknown")
        policy_type = metadata.get("policy_type", "general")
        context_parts.append(f"Policy Document {idx}\nSource: {source},\nPolicy Type: {policy_type}\nContent: {chunk['text']}")

    context = "\n\n".join(context_parts)
    prompt = f"""
    You are a helpful assistant for campus policy related queries.
    Answer the user's question using only the policy context provided below.
    Rules:
    - If the answer is not found in the policy context, respond with "Sorry, I don't have that information."
    - Do not provide any information that is not explicitly stated in the policy context.
    - Keep the answer concise and to the point.

    Policy Context:
    {context}

    User's Question: {query}
    """
    return prompt

# 12. Generate answer using LLM
def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    prompt = build_grounded_prompt(query, retrieved_chunks)
    response = genai_client.models.generate_content(
        model = LLM_MODEL,
        contents = prompt
    )
    return response.text

# 13. Answer the user query using RAG
def answer_question(collection, query: str, top_k:int = 4):
    retrieved_chunks = retrieve_relevant_chunks(collection, query, top_k)
    return generate_answer(query, retrieved_chunks)

def main():
    collection = setup_vector_database()
    folder_path = "./policy_documents"
    build_knowledge_base(collection, folder_path)
    user_query = "Do I need to get my ID card for library access?"
    answer = answer_question(collection, user_query)
    print(f"User Query: {user_query}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

