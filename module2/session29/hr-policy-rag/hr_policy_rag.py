# pip install -U google-genai
from google import genai
import chromadb
from typing import List, Dict
import os

# 1. Initialize GenAI client
genai_client = genai.Client()

# 2. Define Embedding model and LLM model
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.0-flash"

# 3. Sample HR policy documents
POLICY_DOCUMENTS = [
    {
        "id": "hr_doc_1",
        "text": (
            "Employees are entitled to 18 days of annual leave in each calendar year. "
            "Sick leave of up to 8 days may be availed with prior notification to the reporting manager, "
            "and medical documentation may be required for absences longer than two consecutive days. "
            "Unused annual leave up to 5 days may be carried forward to the next calendar year. "
            "Any leave balance beyond the carry-forward limit will lapse at year-end."
        ),
        "metadata": {
            "category": "leave_policy",
            "source": "employee_handbook"
        }
    },
    {
        "id": "hr_doc_2",
        "text": (
            "Eligible employees may work from home for up to 2 days per week based on business requirements. "
            "Employees must complete at least 3 months of service before becoming eligible for regular work-from-home access. "
            "All work-from-home requests must be submitted through the HR portal and approved by the reporting manager in advance. "
            "Employees are expected to remain available during standard working hours while working remotely."
        ),
        "metadata": {
            "category": "work_from_home_policy",
            "source": "remote_work_guidelines"
        }
    },
    {
        "id": "hr_doc_3",
        "text": (
            "The company follows an annual appraisal cycle conducted between April and June each year. "
            "Employee performance is evaluated on a rating scale of 1 to 5, with 5 representing outstanding performance. "
            "Salary increments and performance bonuses are linked to the final appraisal rating and business performance. "
            "Managers are required to complete performance discussions and submit ratings within the defined appraisal timeline."
        ),
        "metadata": {
            "category": "appraisal_policy",
            "source": "performance_management_policy"
        }
    },
    {
        "id": "hr_doc_4",
        "text": (
            "Employees are expected to maintain respectful and professional behavior in all workplace interactions. "
            "Confidential company information, customer data, and internal records must not be shared without proper authorization. "
            "Employees must immediately disclose any conflict of interest that could affect impartial decision-making. "
            "Any violation of workplace conduct or data privacy requirements may result in disciplinary action."
        ),
        "metadata": {
            "category": "code_of_conduct",
            "source": "corporate_ethics_policy"
        }
    }
]

# 4. Create Embeddings
def create_embeddings(text:List[str]) -> List[List[float]]:
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return [item.values for item in response.embeddings]

# 5. Setup Vector Db
def setup_vector_database():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="hr_policy_collection",
        embedding_function=None
    )
    return collection

# 6. Index the documents to vector db
def index_hr_documents(documents, collection):
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    embeddings = create_embeddings(texts)
    collection.upsert(
        ids = ids,
        documents = texts,
        metadatas = metadatas,
        embeddings = embeddings
    )

# 7. Retrieve relevant documents based on query
def retrieve_hr_content(collection, query, top_k=3):
    query_embedding = create_embeddings([query])[0]
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    return results["documents"][0]

# 8. Building prompt for the LLm to generate the final answer
def build_grounded_prompt(query, retrieved_docs):
    context = ""
    num_docs = len(retrieved_docs)
    for idx in range(num_docs):
        context += f"Policy Document {idx+1}:\n{retrieved_docs[idx]}\n\n"
    return f"""
        You are a helpful customer support assistant for ecommerce company.
        Answer the customers's question using only the policy context provided below.
        Rules:
        - If the answer is not found in the policy context, respond with "Sorry, I don't have that information."
        - Do not provide any information that is not explicitly stated in the policy context.
        Keep the answer concise and to the point.
        Mention important policy details and avoid unnecessary information.
        Policy Context: 
        {context}
        Customer Query: 
        {query}

        Answer:
        """
    
# 9. Generate answer using LLM
def generate_answer(query, chunks):
    prompt = build_grounded_prompt(query, chunks)
    response = genai_client.models.generate_content(
        model = LLM_MODEL,
        contents = prompt
    )
    return response.text

# 10. Answer the user query using RAG
def answer_with_rag(collection, query):
    retrieved_docs = retrieve_hr_content(collection, query)
    answer = generate_answer(query, retrieved_docs)
    return answer

# 11. Answer the user query without using RAG
def answer_without_rag(query):
    response = genai_client.models.generate_content(
        model = LLM_MODEL,
        contents = query
    )
    return response.text

def main():
    collection = setup_vector_database()
    index_hr_documents(POLICY_DOCUMENTS, collection)
    user_query = "How many days of annual leave am I entitled to per year?"
    answer1 = answer_with_rag(collection, user_query)
    print(f"Answer with RAG:{answer1}")
    print("\n-----------------------------\n")
    answer2 = answer_without_rag(user_query)
    print(f"Answer without RAG:{answer2}")
   
if __name__ == "__main__":
    main()


