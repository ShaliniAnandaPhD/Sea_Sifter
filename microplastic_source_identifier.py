import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class MicroplasticSourceIdentifier:
    def __init__(self, microplastic_data, source_database):
        """
        Initialize the MicroplasticSourceIdentifier.

        :param microplastic_data: DataFrame containing microplastic sample data.
        :param source_database: DataFrame containing known microplastic source data.

        Required data format:
        - Microplastic sample data:
            - Columns: 'sample_id', 'composition', 'size', 'shape', 'color', 'location'
            - 'composition': Chemical composition of the microplastic (e.g., 'PE', 'PP', 'PS')
            - 'size': Size of the microplastic particle in micrometers (e.g., 100, 500, 1000)
            - 'shape': Shape of the microplastic particle (e.g., 'fiber', 'fragment', 'sphere')
            - 'color': Color of the microplastic particle (e.g., 'red', 'blue', 'transparent')
            - 'location': Geographic coordinates of the sample collection location (e.g., (latitude, longitude))
        - Source database:
            - Columns: 'source_id', 'source_name', 'composition', 'size_range', 'shape', 'color'
            - 'source_name': Name of the microplastic source (e.g., 'Textile fibers', 'Cosmetic microbeads', 'Tire wear')
            - 'composition': Chemical composition of the microplastic from the source
            - 'size_range': Size range of the microplastic particles from the source (e.g., '50-200', '500-1000')
            - 'shape': Typical shape of the microplastic particles from the source
            - 'color': Typical color of the microplastic particles from the source

        Possible errors:
        - KeyError: If the required columns are missing in the data.
        - ValueError: If the data contains invalid or missing values.

        Solutions:
        - Ensure the microplastic sample data and source database have the required columns.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        self.microplastic_data = microplastic_data
        self.source_database = source_database
        self.source_identifier = None

        # Validate the input data
        self._validate_data()

    def _validate_data(self):
        """
        Validate the input data for the microplastic source identification.

        Possible errors:
        - KeyError: If the required columns are missing in the data.
        - ValueError: If the data contains invalid or missing values.

        Solutions:
        - Ensure the microplastic sample data and source database have the required columns.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        # Validate microplastic sample data
        required_columns = ['sample_id', 'composition', 'size', 'shape', 'color', 'location']
        if not all(column in self.microplastic_data.columns for column in required_columns):
            raise KeyError(f"Microplastic sample data is missing required columns. Expected: {required_columns}")

        # Validate source database
        required_columns = ['source_id', 'source_name', 'composition', 'size_range', 'shape', 'color']
        if not all(column in self.source_database.columns for column in required_columns):
            raise KeyError(f"Source database is missing required columns. Expected: {required_columns}")

        # Handle missing values
        self.microplastic_data.dropna(inplace=True)
        self.source_database.dropna(inplace=True)

        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(self.microplastic_data['size']):
            raise ValueError("Microplastic size must be numeric.")

        # Add more data validation checks as needed

    def preprocess_data(self):
        """
        Preprocess the microplastic sample data and source database for source identification.

        - Merge microplastic sample data with source database based on common attributes.
        - Encode categorical variables.
        - Split the data into features and target variable.

        Possible errors:
        - KeyError: If the merging columns are missing in the data.
        - ValueError: If the data contains invalid or inconsistent values.

        Solutions:
        - Ensure the merging columns are present in both microplastic sample data and source database.
        - Handle inconsistencies in categorical variables (e.g., standardize values).
        - Validate the data types and ranges of the preprocessed data.
        """
        # Merge microplastic sample data with source database based on common attributes
        merged_data = pd.merge(self.microplastic_data, self.source_database,
                               on=['composition', 'size', 'shape', 'color'],
                               how='left')

        # Encode categorical variables
        merged_data['composition'] = pd.Categorical(merged_data['composition'])
        merged_data['shape'] = pd.Categorical(merged_data['shape'])
        merged_data['color'] = pd.Categorical(merged_data['color'])
        merged_data['composition_code'] = merged_data['composition'].cat.codes
        merged_data['shape_code'] = merged_data['shape'].cat.codes
        merged_data['color_code'] = merged_data['color'].cat.codes

        # Split the data into features and target variable
        self.X = merged_data[['composition_code', 'size', 'shape_code', 'color_code', 'location']]
        self.y = merged_data['source_name']

    def train_source_identifier(self):
        """
        Train the microplastic source identification model using Random Forest Classifier.

        - Split the data into training and testing sets.
        - Create and train the Random Forest Classifier.
        - Evaluate the model performance using classification report and confusion matrix.

        Possible errors:
        - ValueError: If the input data is not properly preprocessed or has incompatible shapes.

        Solutions:
        - Ensure the data is properly preprocessed using the `preprocess_data` method.
        - Verify that the feature and target variables have compatible shapes.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create and train the Random Forest Classifier
        self.source_identifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.source_identifier.fit(X_train, y_train)

        # Evaluate the model performance
        y_pred = self.source_identifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def identify_sources(self, microplastic_samples):
        """
        Identify the potential sources of microplastic pollution for given microplastic samples.

        :param microplastic_samples: DataFrame containing microplastic sample data for source identification.

        :return: DataFrame with identified sources for each microplastic sample.

        Possible errors:
        - KeyError: If the required columns are missing in the input data.
        - ValueError: If the input data contains invalid or missing values.

        Solutions:
        - Ensure the input data has the same format and columns as the training data.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        # Validate input data
        required_columns = ['sample_id', 'composition', 'size', 'shape', 'color', 'location']
        if not all(column in microplastic_samples.columns for column in required_columns):
            raise KeyError(f"Microplastic sample data is missing required columns. Expected: {required_columns}")

        # Preprocess input data
        preprocessed_samples = microplastic_samples.copy()
        preprocessed_samples['composition'] = pd.Categorical(preprocessed_samples['composition'],
                                                             categories=self.X['composition_code'].cat.categories)
        preprocessed_samples['shape'] = pd.Categorical(preprocessed_samples['shape'],
                                                       categories=self.X['shape_code'].cat.categories)
        preprocessed_samples['color'] = pd.Categorical(preprocessed_samples['color'],
                                                       categories=self.X['color_code'].cat.categories)
        preprocessed_samples['composition_code'] = preprocessed_samples['composition'].cat.codes
        preprocessed_samples['shape_code'] = preprocessed_samples['shape'].cat.codes
        preprocessed_samples['color_code'] = preprocessed_samples['color'].cat.codes

        # Prepare input features
        X_input = preprocessed_samples[['composition_code', 'size', 'shape_code', 'color_code', 'location']]

        # Identify sources using the trained model
        identified_sources = self.source_identifier.predict(X_input)

        # Create a DataFrame with identified sources for each sample
        source_identification_results = pd.DataFrame({
            'sample_id': microplastic_samples['sample_id'],
            'identified_source': identified_sources
        })

        return source_identification_results

    def evaluate_source_contributions(self, source_identification_results):
        """
        Evaluate the contributions of different microplastic sources based on the identification results.

        :param source_identification_results: DataFrame containing source identification results.

        :return: DataFrame with the percentage contribution of each microplastic source.

        Possible errors:
        - ValueError: If the input data is empty or has an inconsistent format.

        Solutions:
        - Ensure the input data is not empty and has the expected columns.
        - Validate the data types and ranges of the input data.
        """
        # Validate input data
        if source_identification_results.empty:
            raise ValueError("Source identification results are empty.")

        # Calculate the percentage contribution of each source
        source_contributions = source_identification_results['identified_source'].value_counts(normalize=True) * 100

        # Create a DataFrame with source contributions
        source_contribution_results = pd.DataFrame({
            'source': source_contributions.index,
            'contribution_percentage': source_contributions.values
        })

        return source_contribution_results

def main():
    # Load microplastic sample data and source database from files or databases
    microplastic_data = pd.read_csv('microplastic_samples.csv')
    source_database = pd.read_csv('source_database.csv')

    # Create an instance of the microplastic source identifier
    source_identifier = MicroplasticSourceIdentifier(microplastic_data, source_database)

    # Preprocess the data
    source_identifier.preprocess_data()

    # Train the source identification model
    source_identifier.train_source_identifier()

    # Identify sources for new microplastic samples
    new_microplastic_samples = pd.read_csv('new_microplastic_samples.csv')
    source_identification_results = source_identifier.identify_sources(new_microplastic_samples)

    # Evaluate the contributions of different microplastic sources
    source_contribution_results = source_identifier.evaluate_source_contributions(source_identification_results)

    # Print the source identification results
    print("Source Identification Results:")
    print(source_identification_results)

    # Print the source contribution results
    print("Source Contribution Results:")
    print(source_contribution_results)

if __name__ == '__main__':
    main()
