import streamlit as st
from chatbot import handle_query
from retriever import get_retriever
from prompt_manager import PromptManager
from vector_store import initialize_vector_store
from data_ingestion import load_data
from langchain_ollama.llms import OllamaLLM
import yaml

# Load configuration
print("Loading configuration...")  # Debugging log
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully!")  # Debugging log
except Exception as e:
    st.error("An error occurred while loading the configuration. Please try again.")
    print(f"Error loading configuration: {e}")

# Load and validate dataset
print("Loading dataset...")  # Debugging log
try:
    file_path = config["data"]["file_path"]
    required_columns = config["data"]["required_columns"]
    data = load_data(file_path, required_columns)
    print(f"Dataset loaded successfully! Number of rows: {len(data)}")  # Debugging log
except Exception as e:
    st.error("An error occurred while loading the dataset. Please try again.")
    print(f"Error loading dataset: {e}")

# Initialize vector store and retriever
print("Initializing vector store and retriever...")  # Debugging log
try:
    db_location = config["vector_store"]["db_location"]
    embedding_model = config["vector_store"]["embedding_model"]
    top_k = config["retriever"]["top_k"]

    vector_store = initialize_vector_store(db_location, embedding_model, data=data)
    print("Vector store initialized successfully!")  # Debugging log

    retriever = get_retriever(vector_store, top_k=top_k)
    print("Retriever initialized successfully!")  # Debugging log
except Exception as e:
    st.error("An error occurred during vector store or retriever initialization. Please try again.")
    print(f"Error during vector store or retriever initialization: {e}")

# Initialize prompt manager
print("Initializing prompt manager...")  # Debugging log
try:
    prompt_template = config["prompt"]["template"]
    prompt_domain = config["prompt"]["domain"]

    prompt_manager = PromptManager(prompt_template, prompt_domain)
    print("Prompt manager initialized successfully!")  # Debugging log
except Exception as e:
    st.error("An error occurred while initializing the prompt manager. Please try again.")
    print(f"Error initializing prompt manager: {e}")

# Initialize language model
print("Initializing language model...")  # Debugging log
try:
    model = OllamaLLM(model="llama3.2")
    print("Language model initialized successfully!")  # Debugging log
except Exception as e:
    st.error("An error occurred while initializing the language model. Please try again.")
    print(f"Error initializing language model: {e}")

# Streamlit chat interface
st.title("AI Customer Support Chatbot")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat interface
st.subheader("Chat with the Assistant")
user_input = st.text_input("Type your question here:", key="user_input")

if st.button("Send"):
    if user_input:
        # Add user input to conversation history
        st.session_state.history.append({"role": "user", "content": user_input})

        # Generate chatbot response
        try:
            response = handle_query(retriever, prompt_manager, model, user_input)
            st.session_state.history.append({"role": "assistant", "content": response})
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"
            st.session_state.history.append({"role": "assistant", "content": response})

# Display conversation history
st.subheader("Conversation")
for message in st.session_state.history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")