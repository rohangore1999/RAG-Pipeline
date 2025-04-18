import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

file_path = Path(__file__).parent / "data" / "somatosensory.pdf"

loader = PyPDFLoader(file_path)

# 1 Data loading
docs = loader.load()

# 2 Chunking func
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# applying chunck fun to docs
split_docs = text_splitter.split_documents(docs)

# 3 Embeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPEN_API_KEY")
)

# connecting to Qdrant vector store (running locally through docker)
# vector_store = QdrantVectorStore.from_documents(
#     documents=[], # for the 1st time it will create
#     url="http://localhost:6333", 
#     collection_name="rag", 
#     embedding=embedder # openai embedder
# )

# # adding document(chunked)
# vector_store.add_documents(documents=split_docs)

print('Injection done')

retriver = QdrantVectorStore.from_documents(
    documents=[], # for the 1st time it will create
    url="http://localhost:6333", 
    collection_name="rag", 
    embedding=embedder # openai embedder
)

# Doining similarity search on user's query
relevant_chunk = retriver.similarity_search(query = "What is Rapidly adapting ??")

# print("Relevant chunk", relevant_chunk)

# Feeding relevant chunk based on similarity seach to model's context
SYSTEM_PROMPT  = f"""
You are an helpful AI Assistant who respond based on the avalable context

Context:
{relevant_chunk}
"""

# Init Openai client
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

user_query = "How do rapidly adapting receptors work?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ],
    temperature=0.7
)

# Print the response
print("\nUser Query:", user_query)
print("\nAssistant Response:", response.choices[0].message.content)