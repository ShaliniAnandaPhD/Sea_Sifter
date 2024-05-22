import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    # Set up OpenAI API key
    # Possible Error: Missing or incorrect API key.
    # Solution: Ensure the API key is correct and properly set in your environment.
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    try:
        # Define the prompt template
        # This template will be used to instruct the language model on what to generate.
        # Possible Error: Template format issues.
        # Solution: Ensure the template string includes the correct placeholders.
        template = """
        Generate a coherent paragraph about the following topic:
        {topic}

        Paragraph:
        """
        prompt = PromptTemplate(
            input_variables=["topic"],  # Define the variables to be filled in the template
            template=template,          # The actual template string
        )

        # Set up the language model with a specified temperature
        # Temperature controls the randomness of the output; 0.7 is moderately random.
        # Possible Error: Issues with API key or model setup.
        # Solution: Ensure the OpenAI API key is valid and parameters are correctly set.
        llm = OpenAI(temperature=0.7)

        # Create the language model chain
        # LLMChain links the language model (llm) with the prompt template (prompt).
        # Possible Error: Incorrect linking of LLM and PromptTemplate.
        # Solution: Ensure that both llm and prompt are correctly instantiated and passed.
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Get user input for the topic
        # This is the topic on which the paragraph will be generated.
        # Possible Error: Empty or invalid input from user.
        # Solution: Add input validation or prompt the user to enter valid information.
        topic = input("Enter the topic for text generation: ")

        # Run the language model chain to generate text
        # The input topic is passed to the chain, which generates the paragraph.
        # Possible Error: Issues with API call or chain execution.
        # Solution: Ensure the chain is correctly set up and the input is valid.
        generated_text = llm_chain.run(topic)

        # Print the generated text
        print("Generated Paragraph:")
        print(generated_text)

    except KeyError as ke:
        # Handle specific KeyError, likely related to missing environment variables.
        print(f"Error: {str(ke)}")
        print("Solution: Ensure the OpenAI API key is properly set in the environment variables.")

    except Exception as e:
        # Handle any other exceptions that may occur.
        print(f"An error occurred: {str(e)}")
        print("Solution: Check the error message and review the code for any issues.")

if __name__ == "__main__":
    # Entry point of the script
    main()
