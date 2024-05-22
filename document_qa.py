import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI

def main():
    # Set up OpenAI API key
    api_key = "your-openai-api-key"
    if not api_key:
        print("Error: OpenAI API key is missing.")
        return
    os.environ["OPENAI_API_KEY"] = api_key

    # Load documents from a directory
    try:
        # Initialize the TextLoader with the directory path and encoding
        loader = TextLoader(directory_path="path/to/documents/directory/", encoding="utf-8")
        # Load documents from the specified directory
        documents = loader.load()
    except FileNotFoundError as e:
        # Possible error: The specified directory does not exist
        print(f"Error: Directory not found. Check the path: {e}")
        return
    except PermissionError as e:
        # Possible error: Insufficient permissions to read the directory
        print(f"Error: Permission denied. Ensure you have the necessary permissions to read the directory: {e}")
        return
    except Exception as e:
        # General error handling for unexpected issues
        print(f"Unexpected error while loading documents: {e}")
        return

    # Create a vector index from the loaded documents
    try:
        # Initialize the VectorstoreIndexCreator with the document loader
        index = VectorstoreIndexCreator().from_loaders([loader])
    except ValueError as e:
        # Possible error: Documents are not in a supported format
        print(f"Error: Failed to create vector index. Ensure documents are in a supported format: {e}")
        return
    except MemoryError as e:
        # Possible error: Insufficient memory to create the vector index
        print(f"Error: Insufficient memory to create vector index: {e}")
        return
    except Exception as e:
        # General error handling for unexpected issues
        print(f"Unexpected error while creating vector index: {e}")
        return

    # Set up the question-answering chain
    try:
        # Initialize the QA chain with the OpenAI model and the retriever from the vector index
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=OpenAI(temperature=0), 
            chain_type="stuff", 
            retriever=index.vectorstore.as_retriever()
        )
    except KeyError as e:
        # Possible error: Invalid chain type or LLM configuration
        print(f"Error: Invalid chain type or LLM configuration: {e}")
        return
    except TypeError as e:
        # Possible error: Incorrect parameters passed to the chain setup
        print(f"Error: Incorrect parameters passed to the QA chain setup: {e}")
        return
    except Exception as e:
        # General error handling for unexpected issues
        print(f"Unexpected error while setting up QA chain: {e}")
        return

    while True:
        # Get user input for the question
        question = input("Enter your question (or 'q' to quit): ").strip()

        if question.lower() == 'q':
            # Exit the loop if the user wants to quit
            break

        # Run the question-answering chain
        try:
            # Pass the user's question to the QA chain and get the result
            result = chain({"question": question})
            # Print the answer and the source documents
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except ValueError as e:
            # Possible error: Invalid input or chain configuration
            print(f"Error: Invalid input or chain configuration: {e}")
        except ConnectionError as e:
            # Possible error: Network issues preventing API call
            print(f"Error: Network issues. Check your internet connection: {e}")
        except KeyError as e:
            # Possible error: Missing expected data in the result
            print(f"Error: Missing data in response. Check the QA chain configuration: {e}")
        except Exception as e:
            # General error handling for unexpected issues
            print(f"Unexpected error during QA process: {e}")

if __name__ == "__main__":
    main()
