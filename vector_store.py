from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd

def initialize_vector_store(db_location, embedding_model, data):
    """
    Initialize the vector store with the given data.

    Args:
        db_location (str): Path to the vector store database.
        embedding_model (str): Name of the embedding model.
        data (pd.DataFrame): The dataset to index.

    Returns:
        Chroma: The initialized vector store.
    """
    print("Step 1.1: Initializing embedding model...")
    # Initialize the embedding model
    embedding = OllamaEmbeddings(model=embedding_model)
    print("Step 1.2: Embedding model initialized successfully!")

    print("Step 1.3: Initializing vector store...")
    # Initialize the Chroma vector store
    vector_store = Chroma(persist_directory=db_location, embedding_function=embedding)
    print("Step 1.4: Vector store initialized successfully!")

    print("Step 1.5: Processing dataset with {} rows...".format(len(data)))
    for start in range(0, len(data), 1000):  # Process in batches of 1000
        end = start + 1000
        batch = data.iloc[start:end]  # Use .iloc to slice the DataFrame

        # Convert the batch to a list of Document objects
        documents = [
            Document(page_content=row["page_content"]) for _, row in batch.iterrows()
        ]
        print(f"Indexing Data Batch: {documents}")

        # Add the documents to the vector store
        vector_store.add_documents(documents)

    print("Step 1.6: Vector store initialization complete.")
    return vector_store