try:
    print("main.py is running...")

    from data_ingestion import load_data
    from vector_store import initialize_vector_store
    from retriever import get_retriever
    from prompt_manager import PromptManager

    print("Starting the application...")

    # Load and validate data
    file_path = "realistic_restaurant_reviews.csv"
    required_columns = {"Title", "Review", "Rating", "Date"}
    print("Loading data...")
    df = load_data(file_path, required_columns)
    print("Data loaded successfully!")

    # Initialize vector store
    db_location = "./chrome_langchain_db"
    embedding_model = "mxbai-embed-large"
    print("Initializing vector store...")
    vector_store = initialize_vector_store(db_location, embedding_model)
    print("Vector store initialized!")

    # Set up retriever
    print("Setting up retriever...")
    retriever = get_retriever(vector_store, top_k=5)
    print("Retriever set up successfully!")

    # Set up prompt manager
    template = """
    You are an expert in answering questions about {domain}.

    Here are some relevant reviews:
    {reviews}

    Based on the reviews, answer the following question concisely and accurately:
    {question}
    """
    prompt_manager = PromptManager(template, domain="a pizza restaurant")
    print("Prompt manager initialized!")

    # Main loop for user interaction
    while True:
        question = input("Ask your question (q to quit): ")
        if question.lower() == "q":
            print("Goodbye!")
            break

        print("Retrieving relevant reviews...")
        reviews = retriever.get_relevant_documents(question)
        reviews_text = "\n".join([doc.page_content for doc in reviews])
        print("Generating prompt...")
        prompt = prompt_manager.generate_prompt(reviews_text, question)
        print("Prompt generated:")
        print(prompt)

except Exception as e:
    print(f"An error occurred: {e}")