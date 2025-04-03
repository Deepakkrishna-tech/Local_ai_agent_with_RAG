import pandas as pd
from vector_store import initialize_vector_store
from retriever import get_retriever  # Ensure this points to the correct module where `get_retriever` is defined

if __name__ == "__main__":
    # Initialize the vector store
    print("Initializing vector store...")
    data = [
        {"page_content": "Refunds can be processed within 7 days of purchase."},
        {"page_content": "To place an order, visit our website and add items to your cart."}
    ]
    data_df = pd.DataFrame(data)  # Convert the list to a Pandas DataFrame

    vector_store = initialize_vector_store(
        db_location="path/to/db",  # Replace with the actual path to your database
        embedding_model="llama-embedding-model",  # Replace with your actual embedding model name
        data=data_df  # Pass the DataFrame instead of a list
    )
    print(f"Vector Store: {vector_store}")

    # Initialize the retriever
    try:
        retriever = get_retriever(vector_store=vector_store, top_k=5)

        # Test the retriever with a sample query
        test_query = "I need a refund"
        print(f"Processing query: {test_query}")
        retrieved_documents = retriever.invoke(test_query)
        print(f"Retrieved Documents: {retrieved_documents}")
    except Exception as e:
        print(f"Error during retriever testing: {e}")