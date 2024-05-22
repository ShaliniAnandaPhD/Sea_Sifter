import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def main():
    # Set up OpenAI API key
    # Possible Error: Missing or incorrect API key.
    # Solution: Ensure the API key is correct and properly set in your environment.
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    try:
        # Load text from a file
        # Possible Error: The file path is incorrect or the file does not exist.
        # Solution: Ensure the file path is correct and the file exists.
        loader = TextLoader("path/to/text/file.txt")
        documents = loader.load()

        # Split the text into chunks
        # Possible Error: Issues with text splitting due to incorrect parameters.
        # Solution: Verify the chunk size and overlap are set correctly.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create embeddings for the text chunks
        # Possible Error: API key issues or incorrect embedding setup.
        # Solution: Ensure the OpenAI API key is valid and embeddings are set up correctly.
        embeddings = OpenAIEmbeddings()

        # Initialize Chroma vector store
        # Possible Error: Issues with initializing the vector store.
        # Solution: Ensure documents are correctly split and embeddings are properly generated.
        db = Chroma.from_documents(texts, embeddings)

        # Define the entity extraction prompt template
        template = """
        Extract named entities from the following text:
        {text}

        Extracted entities:
        """
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )

        # Set up the question-answering chain with entity extraction
        # Possible Error: Issues with setting up the QA chain or incorrect parameters.
        # Solution: Verify the chain type, retriever, and prompt are set up correctly.
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )

        # Get user input for the text to extract entities from
        # Possible Error: User input might be empty or invalid.
        # Solution: Add input validation or prompt the user to enter valid information.
        input_text = input("Enter the text to extract entities from: ")

        # Run entity extraction on the input text
        # Possible Error: Issues with the QA chain or API call failures.
        # Solution: Ensure the QA chain is set up correctly and the API key is valid.
        result = qa_chain.run(input_text)

        # Print the extracted entities
        print("Extracted Entities:")
        print(result)

    except FileNotFoundError:
        # Handle file not found error
        print("Error: The specified text file does not exist.")
        print("Solution: Ensure the file path is correct and the file exists.")

    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred: {str(e)}")
        print("Solution: Check the error message and review the code for any issues.")

if __name__ == "__main__":
    main()
