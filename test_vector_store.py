from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

def test_vector_store():
    print("Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print("Embedding model initialized successfully!")

    print("Initializing vector store...")
    vector_store = Chroma(
        collection_name="test_collection",
        persist_directory="./test_db",
        embedding_function=embeddings
    )
    print("Vector store initialized successfully!")

    # Add a test document
    test_doc = Document(
        page_content="This is a test document.",
        metadata={"category": "test"}
    )
    vector_store.add_documents([test_doc])
    print("Test document added successfully!")

    # Retrieve the document
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    results = retriever.get_relevant_documents("test")
    print(f"Retrieved documents: {results}")

test_vector_store()