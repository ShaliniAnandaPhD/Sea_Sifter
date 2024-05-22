import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pymongo import MongoClient

def get_database():
    """
    Connect to MongoDB and return the database instance.
    Possible Error: Connection issues or incorrect connection string.
    Solution: Ensure the connection string is correct and the MongoDB server is accessible.
    """
    # Replace with your MongoDB connection string and database name
    mongo_connection_string = 'mongodb+srv://username:password@cluster.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['seasifter']

def main():
    # Connect to the MongoDB database
    db = get_database()

    try:
        # Set up OpenAI API key
        # Possible Error: Missing or incorrect API key.
        # Solution: Ensure the API key is correct and properly set in your environment.
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

        # Load research papers from a directory
        # Possible Error: The directory path is incorrect or empty.
        # Solution: Ensure the directory path is correct and contains valid PDF files.
        loader = PyPDFLoader("path/to/research_papers_directory/")
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
        db = Chroma.from_documents(texts, embeddings)

        # Define the summarization prompt template
        # Possible Error: Template format issues.
        # Solution: Ensure the template string includes the correct placeholders.
        template = """
        Synthesize and summarize the key findings and insights from the following research papers related to plastic remediation methods and studies:
        {text}

        Provide a concise summary for each location, focusing on the most effective remediation techniques and their potential impact on reducing microplastic pollution.

        Summary:
        """
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )

        # Set up the summarization chain with the specified language model and prompt
        # Possible Error: Issues with the language model setup.
        # Solution: Ensure the model parameters are correctly set and the prompt is properly linked.
        llm = OpenAI(temperature=0.7)
        summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=prompt)

        # Run the summarization chain on the loaded documents
        # Possible Error: Issues with API call or chain execution.
        # Solution: Ensure the chain is correctly set up and the input is valid.
        summary = summarize_chain({"input_documents": texts}, return_only_outputs=True)

        # Extract the intermediate steps (per-document summaries)
        intermediate_summaries = summary['intermediate_steps']

        # Extract the final synthesized summary
        final_summary = summary['output_text']

        # Store the summaries in MongoDB
        # Possible Error: Issues with MongoDB connection or insertion.
        # Solution: Ensure MongoDB server is running and the connection string is correct.
        db['summaries'].insert_one({
            'intermediate_summaries': intermediate_summaries,
            'final_summary': final_summary
        })

        # Print the final synthesized summary
        print("Synthesized Summary:")
        print(final_summary)

    except FileNotFoundError as fe:
        # Handle file not found error
        print(f"Error: {str(fe)}")
        print("Solution: Ensure the directory path for research papers is correct and contains valid PDF files.")

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
