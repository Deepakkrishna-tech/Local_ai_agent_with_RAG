from typing import List, Union
from retriever import get_retriever
from langchain_ollama.llms import OllamaLLM

def ensure_list(prompts: Union[str, List[str]]) -> List[str]:
    """
    Ensure the input is a list of strings.

    Args:
        prompts (Union[str, List[str]]): The input, which can be a string or a list of strings.

    Returns:
        List[str]: A list of strings.
    """
    if isinstance(prompts, str):
        return [prompts]
    elif isinstance(prompts, list):
        return prompts
    else:
        raise TypeError("Prompts must be a string or a list of strings")

class PromptManager:
    def __init__(self, template, domain):
        """
        Initialize the PromptManager with a template and domain.

        Args:
            template (str): The prompt template.
            domain (str): The domain for the prompt.
        """
        self.template = template
        self.domain = domain
        print("PromptManager initialized with template and domain.")  # Debugging log

    def generate_prompt(self, context, question):
        """
        Generate a prompt based on the template, context, and question.

        Args:
            context (str): The context for the prompt.
            question (str): The user's question.

        Returns:
            str: The generated prompt.
        """
        return self.template.format(domain=self.domain, context=context, question=question)

def handle_query(retriever, prompt_manager, model, prompts):
    """
    Handle the user query by generating a prompt, retrieving context, and generating a response.

    Args:
        retriever: The retriever object for fetching relevant documents.
        prompt_manager: The PromptManager object for generating prompts.
        model: The language model for generating responses.
        prompts (Union[str, List[str]]): The user's input question(s).

    Returns:
        str: The chatbot's response.
    """
    try:
        # Ensure prompts is a list
        prompts = ensure_list(prompts)

        # Retrieve relevant context
        print("Retrieving relevant documents...")
        context_documents = retriever.invoke(prompts[0])  # Use the new invoke method

        # Debugging log for retrieved documents
        print(f"Retrieved Documents: {context_documents}")

        context = "\n".join([doc.page_content for doc in context_documents])

        # Debugging log for context
        print(f"Retrieved Context: {context}")

        # Handle empty context
        if not context:
            context = "No relevant information is available for this query."
            print("No relevant context found. Using default context.")  # Debugging log

        # Generate the prompt
        print("Generating prompt...")
        prompt = prompt_manager.generate_prompt(context, prompts[0])

        # Debugging log for generated prompt
        print(f"Generated Prompt: {prompt}")

        # Generate the response using the language model
        print("Generating response using the language model...")
        response = model.generate([prompt])  # Pass the prompt as a list

        # Debugging log for raw response
        print(f"Raw Response: {response}")

        # Extract the text from the response object
        if isinstance(response, list) and len(response) > 0:
            # Extract the text from the first generation chunk
            if hasattr(response[0], "generations") and len(response[0].generations) > 0:
                return response[0].generations[0].text  # Extract the text content
            elif hasattr(response[0], "text"):
                return response[0].text  # Fallback to 'text' attribute if 'generations' is missing
            else:
                print("Response object does not have a valid 'generations' or 'text' attribute.")  # Debugging log
                return "Sorry, I couldn't generate a response."
        else:
            print("Response is empty or not in the expected format.")  # Debugging log
            return "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error handling query: {e}")
        return f"Error handling query: {e}"

# Test the model with a sample prompt (optional)
if __name__ == "__main__":
    # Initialize a mock model for testing
    model = OllamaLLM(model="llama3.2")  # Replace with your actual model initialization
    test_prompt = "You are a customer support assistant. Answer the following question: Where is my order?"
    response = model.generate([test_prompt])
    print(f"Test Response: {response}")