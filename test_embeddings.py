from langchain_ollama import OllamaEmbeddings

def test_embeddings():
    print("Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print("Embedding model initialized successfully!")

    # Test embedding generation
    test_text = "This is a test sentence."
    embedding = embeddings.embed_query(test_text)
    print(f"Generated embedding for test text: {embedding[:5]}...")  # Print first 5 values

test_embeddings()