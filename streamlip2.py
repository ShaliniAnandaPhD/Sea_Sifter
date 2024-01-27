import streamlit as st
from pymongo import MongoClient
import pandas as pd

# MongoDB connection setup
def get_database():
    # Use your MongoDB connection string and database name
    mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['microplastik']

# Streamlit app
def main():
    st.title("MongoDB Data Visualization")

    # Connect to the database
    db = get_database()

    # Sidebar for user inputs
    st.sidebar.title("Query Parameters")
    ocean = st.sidebar.selectbox("Select Ocean", options=["Pacific Ocean", "Atlantic Ocean", "Indian Ocean", "All"])
    density_class = st.sidebar.selectbox("Select Density Class", options=["High", "Medium", "Low", "All"])
    
    # Query based on user inputs
    query = {}
    if ocean != "All":
        query["Oceans"] = ocean
    if density_class != "All":
        query["Density Class"] = density_class
    
    # Fetch results from MongoDB
    results = db['ncei'].find(query).limit(50)  # Limit to 50 results for performance
    results_df = pd.DataFrame(list(results))

    # Display results
    if not results_df.empty:
        st.write("Query Results:")
        st.dataframe(results_df)
    else:
        st.write("No results found.")

# Run the main function
if __name__ == "__main__":
    main()

