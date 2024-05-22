import os
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def get_database():
    # Replace with your MongoDB connection string and database name
    mongo_connection_string = 'mongodb+srv://username:password@cluster.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['seasifter']

def load_synthesized_papers(directory):
    # Load synthesized research papers from a directory
    loader = PyPDFLoader(directory)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, overlap=0):
    # Split the loaded documents into chunks for better processing
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings(texts):
    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()
    return embeddings

def create_vector_store(texts, embeddings):
    # Create a vector store using Chroma for efficient retrieval
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store

def generate_policy_recommendations(vector_store):
    # Set up the OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # Define the prompt template for generating policy recommendations
    template = """
    Based on the synthesized research papers, generate policy recommendations for governments and organizations to effectively combat microplastic pollution. Consider the following aspects:
    - Key findings and insights from the research papers
    - Identified gaps and limitations in the current research
    - Effective strategies and best practices for mitigating microplastic pollution
    - Potential challenges and barriers to implementation
    - Recommendations for future research and collaborations

    Synthesized Research Papers:
    {context}

    Policy Recommendations:
    """

    prompt = PromptTemplate(
        input_variables=["context"],
        template=template,
    )

    # Set up the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    # Generate policy recommendations
    recommendations = qa_chain.run("Policy recommendations for combating microplastic pollution")
    return recommendations

def store_recommendations(recommendations):
    # Store the generated policy recommendations in MongoDB
    db = get_database()
    result = db["policy_recommendations"].insert_one({"recommendations": recommendations})
    return result.inserted_id

def main():
    try:
        # Load synthesized research papers from a directory
        documents = load_synthesized_papers("path/to/synthesized/papers")

        # Split the documents into chunks
        texts = split_documents(documents)

        # Create embeddings for the text chunks
        embeddings = create_embeddings(texts)

        # Create a vector store using Chroma
        vector_store = create_vector_store(texts, embeddings)

        # Generate policy recommendations
        recommendations = generate_policy_recommendations(vector_store)

        # Store the recommendations in MongoDB
        recommendation_id = store_recommendations(recommendations)

        print("Policy Recommendations:")
        print(recommendations)
        print(f"Recommendations stored in MongoDB with ID: {recommendation_id}")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please make sure the directory path for synthesized papers is correct.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check the error message and traceback for more details.")

if __name__ == "__main__":
    main()
