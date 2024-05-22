import os
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_database():
    # Replace with your MongoDB connection string and database name
    mongo_connection_string = 'mongodb+srv://username:password@cluster.mongodb.net/'
    client = MongoClient(mongo_connection_string)
    return client['seasifter']

def load_data():
    db = get_database()
    
    try:
        # Retrieve data from MongoDB and convert to DataFrame
        data = pd.DataFrame(list(db['impact_data'].find()))
        return data
    except Exception as e:
        print(f"Error: Failed to load data from MongoDB. {str(e)}")
        return None

def preprocess_data(data):
    # Perform data preprocessing steps
    # Example: Handle missing values, encode categorical variables, scale features
    
    # Drop unnecessary columns
    data = data.drop(['_id'], axis=1)
    
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Encode categorical variables if necessary
    # Example: data = pd.get_dummies(data, columns=['category'])
    
    return data

def train_model(data):
    # Split the data into features (X) and target variable (y)
    X = data.drop(['impact_score'], axis=1)
    y = data['impact_score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the machine learning model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    return model

def assess_impact(model, data):
    # Prepare the input data for impact assessment
    input_data = data.drop(['impact_score'], axis=1)
    
    # Make predictions using the trained model
    impact_scores = model.predict(input_data)
    
    # Add the predicted impact scores to the original data
    data['predicted_impact_score'] = impact_scores
    
    return data

def main():
    # Load data from MongoDB
    data = load_data()
    
    if data is not None:
        # Preprocess the data
        preprocessed_data = preprocess_data(data)
        
        # Train the machine learning model
        model = train_model(preprocessed_data)
        
        # Assess the impact of microplastic pollution
        impact_assessment_results = assess_impact(model, preprocessed_data)
        
        # Print the impact assessment results
        print("Impact Assessment Results:")
        print(impact_assessment_results)
    else:
        print("Error: Failed to load data.")

if __name__ == "__main__":
    main()
