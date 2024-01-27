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

# Preprocess the data
# Convert the 'Measurement' column to numeric and handle non-numeric values
data['Measurement'] = pd.to_numeric(data['Measurement'], errors='coerce')

# Drop rows with NaN values in 'Measurement' column (or you could fill them with mean/median)
data = data.dropna(subset=['Measurement'])

# Assuming 'Latitude' and 'Longitude' are your features
# Replace these with the actual column names from your MongoDB collection
X = data[['Latitude', 'Longitude']]
y = data['Measurement']

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
