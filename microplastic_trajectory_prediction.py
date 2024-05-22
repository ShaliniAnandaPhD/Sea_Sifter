import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import folium

class MicroplasticTrajectoryPredictor:
    def __init__(self, data_file):
        """
        Initialize the MicroplasticTrajectoryPredictor.

        :param data_file: Path to the CSV file containing historical microplastic and oceanographic data.

        Possible errors:
        - FileNotFoundError: If the data file is not found.
        - ValueError: If the data file is not a valid CSV file or has missing/invalid data.

        Solutions:
        - Ensure the data file exists at the specified path.
        - Check the data file format and content for any inconsistencies or missing values.
        """
        try:
            self.data = pd.read_csv(data_file)
            self.prepare_data()
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file}")
            raise
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            raise

    def prepare_data(self):
        """
        Prepare the data for training the trajectory prediction model.

        Possible errors:
        - KeyError: If required columns are missing in the data.
        - ValueError: If the data contains invalid values or data types.

        Solutions:
        - Ensure the required columns are present in the data file.
        - Handle missing values and convert data types as needed.
        """
        try:
            # Extract relevant features and target variable
            self.X = self.data[['latitude', 'longitude', 'current_speed', 'current_direction', 'wind_speed', 'wind_direction', 'tide_height']]
            self.y = self.data[['future_latitude', 'future_longitude']]

            # Handle missing values if needed
            self.X.fillna(method='ffill', inplace=True)
            self.y.fillna(method='ffill', inplace=True)

            # Convert data types if needed
            self.X = self.X.astype(float)
            self.y = self.y.astype(float)
        except KeyError as ke:
            print(f"Error: Missing required column(s) in the data: {str(ke)}")
            raise
        except ValueError as ve:
            print(f"Error: Invalid data values or types: {str(ve)}")
            raise

    def train_model(self):
        """
        Train the Random Forest Regressor model for trajectory prediction.

        Possible errors:
        - ValueError: If there is an issue with the input data or model training.

        Solutions:
        - Verify the input data is properly prepared and has the expected format.
        - Adjust the model hyperparameters if needed.
        """
        try:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Create and train the Random Forest Regressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Model training completed. MSE: {mse:.4f}, R2 Score: {r2:.4f}")
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            raise

    def predict_trajectory(self, start_location, num_steps):
        """
        Predict the trajectory of microplastics starting from a given location.

        :param start_location: Dictionary containing the starting latitude and longitude.
        :param num_steps: Number of time steps to predict the trajectory.

        :return: DataFrame containing the predicted trajectory.

        Possible errors:
        - KeyError: If the required keys are missing in the start_location dictionary.
        - ValueError: If the num_steps is not a positive integer.

        Solutions:
        - Ensure the start_location dictionary has the keys 'latitude' and 'longitude'.
        - Provide a positive integer value for num_steps.
        """
        try:
            # Extract the starting latitude and longitude
            start_lat = start_location['latitude']
            start_lon = start_location['longitude']

            # Initialize the trajectory DataFrame
            trajectory = pd.DataFrame(columns=['latitude', 'longitude'])
            trajectory.loc[0] = [start_lat, start_lon]

            # Predict the trajectory for the specified number of steps
            for step in range(1, num_steps + 1):
                # Get the current location and environmental conditions
                current_location = trajectory.iloc[-1]
                current_env_conditions = self.get_env_conditions(current_location)

                # Prepare the input for prediction
                input_data = pd.concat([current_location, current_env_conditions], axis=1)

                # Predict the next location
                next_location = self.model.predict(input_data.values.reshape(1, -1))[0]

                # Append the predicted location to the trajectory
                trajectory.loc[step] = next_location

            return trajectory
        except KeyError as ke:
            print(f"Error: Missing required key(s) in start_location: {str(ke)}")
            raise
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            raise

    def get_env_conditions(self, location):
        """
        Get the environmental conditions at a given location.

        :param location: DataFrame containing the latitude and longitude.

        :return: DataFrame containing the environmental conditions.

        Possible errors:
        - KeyError: If the required keys are missing in the location DataFrame.
        - ValueError: If the location data is invalid or out of range.

        Solutions:
        - Ensure the location DataFrame has the columns 'latitude' and 'longitude'.
        - Verify the location data is within valid ranges.
        """
        try:
            # Extract the latitude and longitude from the location DataFrame
            lat = location['latitude']
            lon = location['longitude']

            # Retrieve the environmental conditions based on latitude and longitude
            # (Placeholder code, replace with actual data retrieval logic)
            env_conditions = pd.DataFrame({
                'current_speed': [0.5],
                'current_direction': [90],
                'wind_speed': [10],
                'wind_direction': [45],
                'tide_height': [1.5]
            })

            return env_conditions
        except KeyError as ke:
            print(f"Error: Missing required column(s) in location DataFrame: {str(ke)}")
            raise
        except ValueError as ve:
            print(f"Error: Invalid location data: {str(ve)}")
            raise

    def visualize_trajectory(self, trajectory, output_file):
        """
        Visualize the predicted microplastic trajectory on a map.

        :param trajectory: DataFrame containing the predicted trajectory.
        :param output_file: Path to save the visualization output file.

        Possible errors:
        - ValueError: If the trajectory data is invalid or empty.
        - IOError: If there is an issue saving the output file.

        Solutions:
        - Ensure the trajectory DataFrame has valid latitude and longitude data.
        - Check the permissions and space availability for saving the output file.
        """
        try:
            # Create a Folium map centered on the starting location
            start_lat = trajectory['latitude'].iloc[0]
            start_lon = trajectory['longitude'].iloc[0]
            m = folium.Map(location=[start_lat, start_lon], zoom_start=6)

            # Add markers for each predicted location
            for idx, row in trajectory.iterrows():
                folium.Marker(location=[row['latitude'], row['longitude']], popup=f"Step {idx}").add_to(m)

            # Add lines connecting the predicted locations
            locations = trajectory[['latitude', 'longitude']].values.tolist()
            folium.PolyLine(locations, color='red', weight=2.5, opacity=1).add_to(m)

            # Save the map as an HTML file
            m.save(output_file)
            print(f"Trajectory visualization saved as {output_file}")
        except ValueError as ve:
            print(f"Error: Invalid trajectory data: {str(ve)}")
            raise
        except IOError as ie:
            print(f"Error: Unable to save the output file: {str(ie)}")
            raise

def main():
    # Set up the data file path
    data_file = "microplastic_data.csv"

    try:
        # Create an instance of the MicroplasticTrajectoryPredictor
        predictor = MicroplasticTrajectoryPredictor(data_file)

        # Train the trajectory prediction model
        predictor.train_model()

        # Set the starting location and number of prediction steps
        start_location = {'latitude': 40.7128, 'longitude': -74.0060}  # Example: New York City
        num_steps = 10

        # Predict the microplastic trajectory
        trajectory = predictor.predict_trajectory(start_location, num_steps)
        print("Predicted Trajectory:")
        print(trajectory)

        # Visualize the predicted trajectory
        output_file = "trajectory_visualization.html"
        predictor.visualize_trajectory(trajectory, output_file)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
