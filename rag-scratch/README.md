# RAG from Scratch

## Overview

RAG combines the power of retrieval systems with generative AI to produce factual, context-aware responses. The system:

1. Ingests documents (PDFs in this case)
2. Breaks them into manageable chunks
3. Creates embeddings (vector representations) of these chunks
4. Stores them in a vector database
5. Retrieves relevant information when queried
6. Uses an LLM to generate answers based on retrieved context

![RAG Flow](flow.png "RAG Flow Diagram")

## Architecture

The pipeline consists of two main parts:

### Data Ingestion Pipeline

- **Data Source**: PDF documents loaded using PyPDFLoader
- **Chunking**: Documents split into smaller sections using RecursiveCharacterTextSplitter
- **Embeddings**: Text chunks converted to vector embeddings using OpenAI's embedding model
- **Vector Store DB**: Embeddings stored in Qdrant vector database

### Query Pipeline

- **User Input**: Question or query from the user
- **Embedding**: Query is converted to the same vector space as the documents
- **Search**: Vector similarity search finds relevant chunks in the database
- **Relevant Chunk**: The most similar content is retrieved
- **LLM**: The relevant chunks are fed to OpenAI's GPT model as context
- **Output**: Model generates a response based on the context and query

## Implementation

This implementation uses:

- LangChain components for document processing and vector operations
- OpenAI embeddings and GPT models for semantic understanding
- Qdrant [from docker] as the vector database for efficient similarity search
