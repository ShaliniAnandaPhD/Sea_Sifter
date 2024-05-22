import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import GraphQAChain
from langchain.llms import OpenAI
from pymongo import MongoClient

def get_database():
    """
    Establish a connection to the MongoDB database used by Seasifter.
    
    Possible Error: Connection issues or incorrect connection string.
    Solution: Ensure the connection string is correct and the MongoDB server is accessible.
    """
    mongo_connection_string = 'mongodb+srv://username:password@cluster.mongodb.net/'
    try:
        client = MongoClient(mongo_connection_string)
        return client['seasifter']
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Solution: Check your MongoDB connection string and ensure the MongoDB server is running.")
        return None

def main():
    # Connect to the MongoDB database
    db = get_database()
    if not db:
        return  # Exit if database connection fails

    try:
        # Set up OpenAI API key
        # Possible Error: Missing or incorrect API key.
        # Solution: Ensure the API key is correct and properly set in your environment.
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

        # Load synthesized research papers from a directory
        # Possible Error: The directory path is incorrect or empty.
        # Solution: Ensure the directory path is correct and contains valid PDF files.
        loader = PyPDFLoader("path/to/synthesized_papers_directory/")
        documents = loader.load()

        # Split the documents into manageable chunks
        # Possible Error: Issues with text splitting due to incorrect parameters.
        # Solution: Verify the chunk size and overlap are set correctly.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create embeddings for the text chunks
        # Possible Error: API key issues or incorrect embedding setup.
        # Solution: Ensure the OpenAI API key is valid and embeddings are set up correctly.
        embeddings = OpenAIEmbeddings()

        # Initialize Chroma vector store for efficient similarity search
        # Possible Error: Issues with initializing the vector store.
        # Solution: Ensure documents are correctly split and embeddings are properly generated.
        vectorstore = Chroma.from_documents(texts, embeddings)

        # Set up the GraphQAChain
        # Possible Error: Issues with the language model setup.
        # Solution: Ensure the model parameters are correctly set and the prompt is properly linked.
        llm = OpenAI(temperature=0.7)
        graph_qa_chain = GraphQAChain.from_llm(llm, graph=vectorstore.similarity_search)

        # Interactive loop for question answering
        while True:
            question = input("Enter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                print("Exiting the knowledge graph builder.")
                break

            # Run the GraphQAChain to answer the question
            # Possible Error: Issues with API call or chain execution.
            # Solution: Ensure the chain is correctly set up and the input is valid.
            try:
                answer = graph_qa_chain.run(question)
            except Exception as e:
                print(f"Error answering question: {e}")
                print("Solution: Ensure the question is correctly formatted and the API key is valid.")
                continue

            # Print the answer
            print("Answer:", answer)

            # Store the question and answer in MongoDB
            # Possible Error: Issues with MongoDB connection or insertion.
            # Solution: Ensure MongoDB server is running and the connection string is correct.
            try:
                db['knowledge_graph'].insert_one({
                    'question': question,
                    'answer': answer
                })
            except Exception as e:
                print(f"Error storing question and answer in MongoDB: {e}")
                print("Solution: Check MongoDB server status and connection details.")

    except FileNotFoundError as fe:
        print(f"Error: {str(fe)}")
        print("Solution: Ensure the directory path for synthesized research papers is correct and contains valid PDF files.")

    except KeyError as ke:
        print(f"Error: {str(ke)}")
        print("Solution: Ensure the OpenAI API key is properly set in the environment variables.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Solution: Check the error message and review the code for any issues.")

if __name__ == "__main__":
    main()
