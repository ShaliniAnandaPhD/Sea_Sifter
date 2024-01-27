import pandas as pd
from pymongo import MongoClient
import folium

# Connect to MongoDB and fetch data
def get_database():
    mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['microplastik']

def get_marker_color(category):
    colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red',
        'Very High': 'darkred'
    }
    return colors.get(category, 'gray')

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
    db = get_database()
    collection = db['microplastik.ncei']
    data = pd.DataFrame(list(collection.find()))

    if 'Latitude' not in data.columns or 'Longitude' not in data.columns or 'Density Class' not in data.columns:
        print("Required columns are not found in the data.")
        return

    map_display = create_map(data)
    map_display.save('microplastic_density_map.html')
    print("Map with density data has been saved to microplastic_density_map.html")

if __name__ == "__main__":
    main()



