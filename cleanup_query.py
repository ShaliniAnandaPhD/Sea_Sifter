from pymongo import MongoClient

# Connect to MongoDB using the connection string
mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
client = MongoClient(mongo_connection_string)
db = client.microplastik

# Function to query microplastic densities in a given location
def get_microplastic_density(location):
    densities = db.densities.find({"location": location})
    return list(densities)

# Function to get cleanup methods for a given density
def get_cleanup_methods(density):
    methods = db.cleanup_methods.find({"density": density})
    return list(methods)

# Main usage example
def main():
    # Define the location you want to query
    location = "San Francisco Bay"

    # Get microplastic density data for the specified location
    try:
        density_data = get_microplastic_density(location)

        # Iterate through each density data point
        for density in density_data:
            # Get cleanup methods for the specific density
            methods = get_cleanup_methods(density['density'])

            # Print the cleanup methods for each density in the location
            print(f"Cleanup methods for density '{density['density']}' in {location}: {methods}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()

