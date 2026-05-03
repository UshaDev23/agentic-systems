# agentic-systems
The main aim of this project is to create a RAG pipeline for generating more relevant responses by retrieval augmented approach. In this approach first we are storing the poliy details of the company within the vector by converting the documents into embeddings. When the user asks the query it is again converted into a vector embedding whose embeddings are compared with the existing policies and the matching records are used as a context. This context is passed to the model so that it generates the reponses using that specific context only rather than generating some generic responses.

To run the project first we need to install the following:-
python3 -m venv venv --> To setup the virtual environment
venv\Scripts\activate --> To activate the environment
pip install chromaDB --> Install vector Db
pip install -U google-genai --> google Gemini
Store the API_KEY within the User Variables within the environment variables of your system 

After the above process is completed we can just execute the main function using the following commnad python hr_policy_rag.py 
