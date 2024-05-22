import os
from rasa.core.agent import Agent
from pymongo import MongoClient

def get_database():
    """
    Connect to MongoDB and return the database instance.
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

def load_agent():
    """
    Load the trained Rasa model.
    Possible Error: Incorrect model path or missing model files.
    Solution: Ensure the model path is correct and the trained model files are present.
    """
    model_path = "path/to/rasa/model"
    try:
        agent = Agent.load(model_path)
        return agent
    except Exception as e:
        print(f"Error loading Rasa model: {e}")
        print("Solution: Check the model path and ensure the trained model files are available.")
        return None

def process_user_message(agent, message):
    """
    Process user message using the Rasa agent.
    Possible Error: Issues with the agent handling the text.
    Solution: Ensure the Rasa model is properly trained and configured to handle the inputs.
    """
    try:
        response = agent.handle_text(message)
        return response
    except Exception as e:
        print(f"Error processing user message: {e}")
        print("Solution: Review the Rasa model training and configuration.")
        return []

def get_microplastic_data(location):
    """
    Retrieve microplastic data for a specific location from MongoDB.
    Possible Error: Issues with data retrieval or missing data for the location.
    Solution: Ensure the location data is present in the MongoDB collection.
    """
    db = get_database()
    if db:
        try:
            data = db['microplastic_data'].find_one({'location': location})
            return data
        except Exception as e:
            print(f"Error retrieving microplastic data: {e}")
            print("Solution: Check the MongoDB query and ensure the data is correctly stored.")
            return None
    return None

def get_remediation_methods(location):
    """
    Retrieve remediation methods for a specific location from MongoDB.
    Possible Error: Issues with data retrieval or missing data for the location.
    Solution: Ensure the remediation methods data is present in the MongoDB collection.
    """
    db = get_database()
    if db:
        try:
            methods = db['remediation_methods'].find_one({'location': location})
            return methods
        except Exception as e:
            print(f"Error retrieving remediation methods: {e}")
            print("Solution: Check the MongoDB query and ensure the data is correctly stored.")
            return None
    return None

def main():
    """
    Main function to run the chatbot interface.
    """
    # Load the Rasa agent
    agent = load_agent()
    if not agent:
        print("Failed to load the Rasa agent. Exiting the chatbot interface.")
        return

    print("Welcome to the Seasifter Chatbot!")
    print("How can I assist you with information about microplastic pollution?")

    while True:
        user_message = input("User: ")

        if user_message.lower() == 'quit':
            print("Thank you for using the Seasifter Chatbot. Goodbye!")
            break

        response = process_user_message(agent, user_message)

        if response:
            for message in response:
                print("Chatbot:", message['text'])

                if 'location' in message:
                    location = message['location']
                    microplastic_data = get_microplastic_data(location)
                    remediation_methods = get_remediation_methods(location)

                    if microplastic_data:
                        print("Microplastic Data for", location)
                        print("Density:", microplastic_data.get('density', 'N/A'))
                        print("Common Types:", microplastic_data.get('common_types', 'N/A'))
                        # Print other relevant data fields

                    if remediation_methods:
                        print("Remediation Methods for", location)
                        print("Methods:", remediation_methods.get('methods', 'N/A'))
                        print("Effectiveness:", remediation_methods.get('effectiveness', 'N/A'))
                        # Print other relevant data fields

if __name__ == "__main__":
    main()
