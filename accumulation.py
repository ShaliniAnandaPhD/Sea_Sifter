import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Connect to MongoDB and fetch data
def get_database():
    mongo_connection_string = 'mongodb+srv://nlpvisionio:1Khi70ddpq1Aldg8@microplastik.mz9kfj6.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['microplastik']

# Assuming the 'microplastik.ncei' collection has the required data
db = get_database()
collection = db['microplastik.ncei']
data = pd.DataFrame(list(collection.find()))

# Check if the data is empty
if data.empty:
    print("No data found in the database.")
else:
    # Assuming 'latitude', 'longitude', and other environmental factors are your features
    # and 'microplastic_density' (or similar) is your target variable
    # Replace these with the actual column names from your MongoDB collection

    # Split the data into features and target variable
    X = data[['latitude', 'longitude', 'environmental_factor1', 'environmental_factor2']]  # Replace with actual features
    y = data['microplastic_density']  # Replace with actual target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

