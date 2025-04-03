import pandas as pd
from vector_store import initialize_vector_store

def get_retriever(vector_store, top_k=5):
    """
    Initialize the retriever from the vector store.

    Args:
        vector_store: The initialized vector store.
        top_k (int): Number of top results to retrieve.

    Returns:
        Retriever: The retriever instance.
    """
    try:
        if vector_store is None:
            raise ValueError("Vector store is not initialized. Please initialize it before calling get_retriever.")

        print("Initializing retriever...")
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        print("Retriever initialized successfully.")
        return retriever
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise


if __name__ == "__main__":
    # Initialize the vector store
    print("Initializing vector store...")
    data = [
        {"page_content": "Refunds can be processed within 7 days of purchase."},
        {"page_content": "To place an order, visit our website and add items to your cart."}
    ]

    try:
        from vector_store import initialize_vector_store

        data_df = pd.DataFrame(data)  # Convert the list to a Pandas DataFrame

        vector_store = initialize_vector_store(
            db_location="path/to/db",  # Replace with the actual path to your database
            embedding_model="mxbai-embed-large:latest",  # Use the valid embedding model name
            data=data_df  # Pass the DataFrame instead of a list
        )
        print(f"Vector Store: {vector_store}")

        # Initialize the retriever
        retriever = get_retriever(vector_store=vector_store, top_k=5)
        print(f"Retriever initialized: {retriever}")

        # Test the retriever with a sample query
        test_query = "I need a refund"
        print(f"Processing query: {test_query}")
        retrieved_documents = retriever.invoke(test_query)
        print(f"Retrieved Documents: {retrieved_documents}")

    except Exception as e:
        print(f"Error during retriever testing: {e}")