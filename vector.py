from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the CSV file with error handling
try:
    df = pd.read_csv("realistic_restaurant_reviews.csv")
except FileNotFoundError:
    raise FileNotFoundError("The file 'realistic_restaurant_reviews.csv' was not found.")
except pd.errors.EmptyDataError:
    raise ValueError("The file 'realistic_restaurant_reviews.csv' is empty or invalid.")

# Validate required columns
required_columns = {"Title", "Review", "Rating", "Date"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Initialize embeddings
embedding_model = "mxbai-embed-large"
embeddings = OllamaEmbeddings(model=embedding_model)

# Set up the database location
db_location = os.path.join(".", "chrome_langchain_db")
add_documents = not os.path.exists(db_location)

# Add documents to the vector store if needed
if add_documents:
    logger.info("Adding documents to the vector store...")
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Initialize the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    logger.info("Documents added successfully.")

# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
logger.info("Retriever is ready for use.")