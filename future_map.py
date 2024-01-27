import pandas as pd
import streamlit as st
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static

# Connect to MongoDB and fetch data
def get_database():
    mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['microplastik']

def get_marker_color(category):
    """
    Determines the color of a marker based on the microplastic category.
    :param category: The microplastic category.
    :return: A string representing the color.
    """
    colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red',
        'Very High': 'darkred'  # Using 'darkred' for 'Very High' category
    }
    return colors.get(category, 'gray')  # Default color if category is not found

def create_map(data):
    avg_lat = data['Latitude'].mean() if 'Latitude' in data else 0
    avg_lng = data['Longitude'].mean() if 'Longitude' in data else 0
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=2)

    for index, row in data.iterrows():
        if 'Latitude' in row and 'Longitude' in row and 'Density Class' in row:
            color = get_marker_color(row['Density Class'])
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=f"Density Class: {row['Density Class']}",
                icon=folium.Icon(color=color)
            ).add_to(m)
    return m

def main():
    st.title("Microplastic Accumulation Map")

    db = get_database()
    collection = db['microplastik.ncei']
    data = pd.DataFrame(list(collection.find()))

    # Assuming 'Latitude', 'Longitude', and 'Density Class' are columns in your data
    if 'Latitude' not in data.columns or 'Longitude' not in data.columns or 'Density Class' not in data.columns:
        st.write("Required columns are not found in the data.")
        return

    # Map Visualization
    map_display = create_map(data)
    folium_static(map_display)

if __name__ == "__main__":
    main()
