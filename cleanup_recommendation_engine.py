import os
from pymongo import MongoClient
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_database():
    """
    Establish a connection to the MongoDB database used by Seasifter.
    
    Possible Error: Connection issues or incorrect connection string.
    Solution: Ensure the connection string is correct and the MongoDB server is accessible.
    """
    # Replace with your MongoDB connection string and database name
    mongo_connection_string = 'mongodb+srv://username:password@cluster.mongodb.net/'
    try:
        client = MongoClient(mongo_connection_string)
        return client['seasifter']
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Solution: Check your MongoDB connection string and ensure the MongoDB server is running.")
        return None

def get_location_data(location):
    """
    Retrieve location-specific data from the MongoDB collection 'location_data' based on the provided location.
    
    Possible Error: Issues with data retrieval or missing data for the location.
    Solution: Ensure the location data is present in the MongoDB collection and the query is correct.
    """
    db = get_database()
    if db:
        try:
            data = db['location_data'].find_one({'location': location})
            if data:
                return data
            else:
                print(f"No data found for location '{location}'.")
                return None
        except Exception as e:
            print(f"Error: Failed to retrieve location data from MongoDB. {str(e)}")
            return None
    else:
        return None

def get_remediation_methods():
    """
    Retrieve remediation methods data from the MongoDB collection 'remediation_methods'.
    
    Possible Error: Issues with data retrieval or missing data in the collection.
    Solution: Ensure the remediation methods data is present in the MongoDB collection and the query is correct.
    """
    db = get_database()
    if db:
        try:
            methods = list(db['remediation_methods'].find())
            if methods:
                return methods
            else:
                print("No remediation methods found.")
                return None
        except Exception as e:
            print(f"Error: Failed to retrieve remediation methods from MongoDB. {str(e)}")
            return None
    else:
        return None

def generate_recommendation(location_data, remediation_methods):
    """
    Generate a clean-up recommendation using OpenAI's language model based on the provided location data and remediation methods.
    
    Possible Error: Issues with the OpenAI API or the prompt template.
    Solution: Ensure the OpenAI API key is valid, and the prompt template is correctly formatted.
    """
    # Set up OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    # Define the prompt template for generating recommendations
    template = """
    Based on the following location data and available remediation methods, generate a recommendation for the most effective clean-up strategy and technology for the given location.

    Location Data:
    {location_data}

    Remediation Methods:
    {remediation_methods}

    Recommendation:
    """

    prompt = PromptTemplate(
        input_variables=["location_data", "remediation_methods"],
        template=template,
    )

    # Set up the language model and recommendation chain
    llm = OpenAI(temperature=0.7)
    recommendation_chain = LLMChain(llm=llm, prompt=prompt)

    try:
        # Generate the recommendation using the LLMChain
        recommendation = recommendation_chain.run(location_data=location_data, remediation_methods=remediation_methods)
        return recommendation
    except Exception as e:
        print(f"Error: Failed to generate recommendation. {str(e)}")
        return None

def main():
    location = input("Enter the location: ")
    
    # Retrieve location-specific data from MongoDB
    location_data = get_location_data(location)
    
    if location_data:
        # Retrieve remediation methods data from MongoDB
        remediation_methods = get_remediation_methods()
        
        if remediation_methods:
            # Generate cleanup recommendation using the LLMChain
            recommendation = generate_recommendation(location_data, remediation_methods)
            
            if recommendation:
                print("Clean-up Recommendation:")
                print(recommendation)
            else:
                print("Error: Failed to generate recommendation.")
        else:
            print("Error: Failed to retrieve remediation methods.")
    else:
        print(f"Error: No data found for location '{location}'.")

if __name__ == "__main__":
    main()
