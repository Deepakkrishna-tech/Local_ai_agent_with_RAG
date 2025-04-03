import streamlit as st

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
        print("PromptManager initialized with template and domain.")

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