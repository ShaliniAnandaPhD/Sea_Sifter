import pandas as pd
from prophet import Prophet
from gluonts.dataset.common import ListDataset
from gluonts.model.prophet import ProphetPredictor
from gluonts.evaluation.backtest import make_evaluation_predictions

class AdvancedMicroplasticForecasting:
    def __init__(self, data_path):
        """
        Initialize the AdvancedMicroplasticForecasting class.

        Args:
            data_path (str): The path to the microplastic accumulation data file (CSV).
        """
        self.data_path = data_path
        self.data = None
        self.prophet_model = None
        self.gluonts_model = None

    def load_data(self):
        """
        Load the microplastic accumulation data from the specified file.

        Possible Errors:
        - File not found: Ensure that the data file exists at the specified path.
        - Invalid file format: Verify that the data file is in the correct CSV format.

        Solutions:
        - Double-check the data_path and make sure it points to the correct file.
        - Ensure that the data file is a valid CSV file with the required columns.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            # Rename columns to match Prophet's requirements
            self.data.rename(columns={'date': 'ds', 'accumulation': 'y'}, inplace=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def train_prophet_model(self):
        """
        Train the Facebook Prophet model on the microplastic accumulation data.

        Possible Errors:
        - Missing required columns: Ensure that the data has the required 'ds' and 'y' columns.
        - Incorrect data format: Verify that the 'ds' column contains valid date strings and 'y' column contains numeric values.

        Solutions:
        - Rename the columns in the data to match Prophet's requirements ('ds' for date and 'y' for accumulation).
        - Ensure that the date column is in a valid format (e.g., 'YYYY-MM-DD') and the accumulation column contains numeric values.
        """
        try:
            self.prophet_model = Prophet()
            self.prophet_model.fit(self.data)
        except KeyError as e:
            print(f"Error: Missing required column in data: {e}")
            raise
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            raise

    def predict_with_prophet(self, future_periods):
        """
        Make predictions using the trained Facebook Prophet model.

        Args:
            future_periods (int): The number of future periods to predict.

        Returns:
            pd.DataFrame: The predicted microplastic accumulation values.

        Possible Errors:
        - Model not trained: Ensure that the Prophet model is trained before making predictions.
        - Invalid future_periods: Verify that future_periods is a positive integer.

        Solutions:
        - Train the Prophet model using the `train_prophet_model` method before making predictions.
        - Ensure that future_periods is a valid positive integer value.
        """
        try:
            future_dates = self.prophet_model.make_future_dataframe(periods=future_periods)
            forecast = self.prophet_model.predict(future_dates)
            return forecast[['ds', 'yhat']]
        except AttributeError:
            print("Error: Prophet model not trained. Train the model before making predictions.")
            raise
        except Exception as e:
            print(f"Error making predictions with Prophet: {e}")
            raise

    def train_gluonts_model(self):
        """
        Train the GluonTS model on the microplastic accumulation data.

        Possible Errors:
        - Invalid data format: Ensure that the data is in the correct format expected by GluonTS.
        - Incorrect data frequency: Verify that the data frequency is specified correctly (e.g., 'D' for daily, 'M' for monthly).

        Solutions:
        - Preprocess the data to match the format expected by GluonTS (time index and accumulation values).
        - Specify the correct frequency of the data when creating the ListDataset.
        """
        try:
            # Preprocess the data for GluonTS
            gluonts_data = ListDataset(
                [{"start": self.data['ds'].min(), "target": self.data['y'].values}],
                freq='D'  # Specify the frequency of the data (e.g., 'D' for daily, 'M' for monthly)
            )

            # Create and train the GluonTS ProphetPredictor
            self.gluonts_model = ProphetPredictor(prediction_length=30, freq='D')
            self.gluonts_model.train(gluonts_data)
        except Exception as e:
            print(f"Error training GluonTS model: {e}")
            raise

    def predict_with_gluonts(self, forecast_horizon):
        """
        Make predictions using the trained GluonTS model.

        Args:
            forecast_horizon (int): The number of periods to forecast.

        Returns:
            pd.DataFrame: The predicted microplastic accumulation values.

        Possible Errors:
        - Model not trained: Ensure that the GluonTS model is trained before making predictions.
        - Invalid forecast_horizon: Verify that forecast_horizon is a positive integer.

        Solutions:
        - Train the GluonTS model using the `train_gluonts_model` method before making predictions.
        - Ensure that forecast_horizon is a valid positive integer value.
        """
        try:
            # Create the ListDataset for prediction
            gluonts_data = ListDataset(
                [{"start": self.data['ds'].min(), "target": self.data['y'].values}],
                freq='D'  # Specify the frequency of the data (e.g., 'D' for daily, 'M' for monthly)
            )

            # Make predictions using the trained GluonTS model
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=gluonts_data,
                predictor=self.gluonts_model,
                num_samples=100
            )

            # Convert the predictions to a pandas DataFrame
            predictions = list(forecast_it)[0].mean
            forecast_dates = pd.date_range(start=self.data['ds'].max(), periods=forecast_horizon + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': predictions})
            return forecast_df
        except AttributeError:
            print("Error: GluonTS model not trained. Train the model before making predictions.")
            raise
        except Exception as e:
            print(f"Error making predictions with GluonTS: {e}")
            raise

def main():
    # Set the path to the microplastic accumulation data file
    data_path = "path/to/your/microplastic_data.csv"

    # Initialize the AdvancedMicroplasticForecasting instance
    forecaster = AdvancedMicroplasticForecasting(data_path)

    # Load the microplastic accumulation data
    forecaster.load_data()

    # Train the Facebook Prophet model
    forecaster.train_prophet_model()

    # Make predictions using Facebook Prophet
    prophet_predictions = forecaster.predict_with_prophet(future_periods=30)
    print("Facebook Prophet Predictions:")
    print(prophet_predictions)

    # Train the GluonTS model
    forecaster.train_gluonts_model()

    # Make predictions using GluonTS
    gluonts_predictions = forecaster.predict_with_gluonts(forecast_horizon=30)
    print("GluonTS Predictions:")
    print(gluonts_predictions)

if __name__ == "__main__":
    main()
