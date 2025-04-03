import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def load_data(file_path, required_columns):
    """
    Load and validate the customer support data.

    Args:
        file_path (str): Path to the data file.
        required_columns (list): List of required column names.

    Returns:
        pd.DataFrame: A DataFrame containing the validated data.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is empty or missing required columns.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty or invalid.")

    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"The data file must contain the following columns: {required_columns}")
    
    # Debugging log for loaded data
    print(f"Loaded Data: {df}")

    # Debugging log for data format
    print(f"Data Format: {type(df)}")
    
    return df

def initialize_vector_store(db_location, embedding_model, documents=None, ids=None):
    """
    Initialize the vector store for RAG.

    Args:
        db_location (str): Path to the database directory.
        embedding_model (str): Name of the embedding model.
        documents (list, optional): List of documents to add to the vector store.
        ids (list, optional): List of IDs corresponding to the documents.

    Returns:
        Chroma: The initialized vector store.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name="customer_support",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    if documents and ids:
        vector_store.add_documents(documents=documents, ids=ids)
    return vector_store