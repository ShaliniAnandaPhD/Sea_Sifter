from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

def main():
    # Set up OpenAI API key
    # Possible error: If the API key is missing or incorrect, the OpenAI service will not authenticate.
    # Solution: Ensure the API key is correct and properly set.
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    # Define the custom prompt template with placeholders for user input
    template = """
    Please write a {text_type} about {topic}.

    Include the following keywords in the {text_type}: {keywords}.

    The {text_type} should be approximately {length} words long.
    """

    # Create a PromptTemplate instance
    # Possible error: If the template is incorrectly formatted or missing placeholders,
    # the PromptTemplate instance may not be created correctly.
    # Solution: Ensure the template string includes all necessary placeholders.
    prompt = PromptTemplate(
        input_variables=["text_type", "topic", "keywords", "length"],
        template=template,
    )

    # Set up the OpenAI language model
    # Possible error: If the model setup parameters are incorrect, the model may not perform as expected.
    # Solution: Ensure all parameters (like temperature) are set correctly.
    llm = OpenAI(temperature=0.7)

    # Get user input for the prompt variables
    # Possible error: User input might be empty or invalid.
    # Solution: Add input validation or prompt the user to enter valid information.
    text_type = input("Enter the type of text to generate (e.g., article, story): ")
    topic = input("Enter the topic: ")
    keywords = input("Enter the keywords to include (comma-separated): ")
    length = input("Enter the approximate length in words: ")

    # Format the prompt with user input
    # Possible error: If the user input does not match the expected format,
    # the formatted prompt may be incorrect.
    # Solution: Validate and preprocess user input as necessary.
    formatted_prompt = prompt.format(
        text_type=text_type,
        topic=topic,
        keywords=keywords,
        length=length,
    )

    # Generate text using the language model and formatted prompt
    # Possible error: If the OpenAI API key is invalid or there are network issues,
    # the language model may fail to generate text.
    # Solution: Ensure the OpenAI API key is valid and check your internet connection.
    try:
        generated_text = llm(formatted_prompt)
    except Exception as e:
        # Handle potential errors during text generation
        print(f"Error generating text: {e}")
        return

    # Print the generated text
    print(generated_text)

if __name__ == "__main__":
    main()
