# Local AI Agent with Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)** AI agent for customer support. It combines a vector store for document retrieval and a language model for response generation, providing accurate and context-aware responses.

---

## **System Architecture**

### **High-Level Architecture**
The system consists of the following components:
1. **Data Ingestion Module**: Prepares and processes the dataset for indexing.
2. **Vector Store**: Stores embeddings of documents for efficient similarity-based retrieval.
3. **Retriever**: Fetches the most relevant documents from the vector store based on user queries.
4. **Language Model**: Generates responses based on retrieved documents and user queries.
5. **User Interface**: A Streamlit-based web application for user interaction.

---

## **Components**

### **1. Data Ingestion Module**
- **Purpose**: Prepares the dataset for indexing in the vector store.
- **Input**: A CSV file containing customer support documents.
- **Output**: A processed dataset ready for embedding.

### **2. Vector Store**
- **Purpose**: Stores document embeddings for similarity-based retrieval.
- **Technology**: Chroma (a vector database).

### **3. Retriever**
- **Purpose**: Fetches the most relevant documents from the vector store based on user queries.
- **Key Features**:
  - Uses similarity search to retrieve top-k documents.

### **4. Language Model**
- **Purpose**: Generates responses based on retrieved documents and user queries.
- **Technology**: Pre-trained language models (e.g., `mxbai-embed-large:latest`).

### **5. User Interface**
- **Purpose**: Provides an interactive interface for users to interact with the chatbot.
- **Technology**: Streamlit.

---![ChatGPT Image Apr 3, 2025, 05_50_20 PM](https://github.com/user-attachments/assets/28905a0f-3e88-4d13-8536-a1045bd984a5)


## **Workflow**

### **1. Initialization**
1. Load the dataset using the **Data Ingestion Module**.
2. Initialize the **Vector Store** and embed the documents.
3. Initialize the **Retriever** with the vector store.

### **2. Query Handling**
1. The user enters a query in the **Streamlit User Interface**.
2. The **Retriever** fetches the most relevant documents from the vector store.
3. The **Language Model** generates a response based on the query and retrieved documents.
4. The response is displayed to the user in the **Streamlit User Interface**.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - `langchain`: For vector store and retriever integration.
  - `streamlit`: For building the user interface.
  - `pandas`: For data processing.
  - `requests`: For downloading files from Google Drive.
- **External Tools**:
  - **Google Drive**: For hosting large files (e.g., `chroma.sqlite3`).

---

## **Setup and Deployment**

### **1. Local Deployment**
1. Clone the repository:
   ```bash
   git clone https://github.com/Deepakkrishna-tech/Local_ai_agent_with_RAG.git
   cd Local_ai_agent_with_RAG
