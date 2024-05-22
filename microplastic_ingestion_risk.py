import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class MicroplasticIngestionRiskAssessment:
    def __init__(self, microplastic_data, species_data):
        """
        Initialize the MicroplasticIngestionRiskAssessment tool.

        :param microplastic_data: DataFrame containing microplastic data.
        :param species_data: DataFrame containing marine species data.

        Required data format:
        - Microplastic data:
            - Columns: 'id', 'size', 'shape', 'composition', 'concentration'
            - 'size': Microplastic size in micrometers (e.g., 100, 500, 1000)
            - 'shape': Microplastic shape (e.g., 'fiber', 'fragment', 'sphere')
            - 'composition': Chemical composition of the microplastic (e.g., 'PE', 'PP', 'PS')
            - 'concentration': Concentration of microplastics in the environment (particles/m^3)
        - Species data:
            - Columns: 'species', 'feeding_habit', 'habitat', 'size_range'
            - 'species': Name of the marine species (e.g., 'Cod', 'Shrimp', 'Mussel')
            - 'feeding_habit': Feeding habit of the species (e.g., 'filter_feeder', 'predator', 'grazer')
            - 'habitat': Habitat preference of the species (e.g., 'pelagic', 'benthic', 'coastal')
            - 'size_range': Size range of the species in centimeters (e.g., '10-20', '30-50')

        Possible errors:
        - KeyError: If the required columns are missing in the data.
        - ValueError: If the data contains invalid or missing values.

        Solutions:
        - Ensure the microplastic and species data have the required columns.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        self.microplastic_data = microplastic_data
        self.species_data = species_data
        self.risk_model = None

        # Validate the input data
        self._validate_data()

    def _validate_data(self):
        """
        Validate the input data for the risk assessment.

        Possible errors:
        - KeyError: If the required columns are missing in the data.
        - ValueError: If the data contains invalid or missing values.

        Solutions:
        - Ensure the microplastic and species data have the required columns.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        # Validate microplastic data
        required_columns = ['id', 'size', 'shape', 'composition', 'concentration']
        if not all(column in self.microplastic_data.columns for column in required_columns):
            raise KeyError(f"Microplastic data is missing required columns. Expected: {required_columns}")
        
        # Validate species data
        required_columns = ['species', 'feeding_habit', 'habitat', 'size_range']
        if not all(column in self.species_data.columns for column in required_columns):
            raise KeyError(f"Species data is missing required columns. Expected: {required_columns}")
        
        # Handle missing values
        self.microplastic_data.dropna(inplace=True)
        self.species_data.dropna(inplace=True)
        
        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(self.microplastic_data['size']):
            raise ValueError("Microplastic size must be numeric.")
        
        if not pd.api.types.is_numeric_dtype(self.microplastic_data['concentration']):
            raise ValueError("Microplastic concentration must be numeric.")
        
        # Add more data validation checks as needed
        
    def preprocess_data(self):
        """
        Preprocess the microplastic and species data for risk assessment.
        
        - Merge microplastic and species data based on common attributes.
        - Encode categorical variables.
        - Split the data into features and target variable.
        
        Possible errors:
        - KeyError: If the merging columns are missing in the data.
        - ValueError: If the data contains invalid or inconsistent values.
        
        Solutions:
        - Ensure the merging columns are present in both microplastic and species data.
        - Handle inconsistencies in categorical variables (e.g., standardize values).
        - Validate the data types and ranges of the preprocessed data.
        """
        # Merge microplastic and species data based on common attributes
        merged_data = pd.merge(self.microplastic_data, self.species_data, on=['size', 'shape', 'composition'])
        
        # Encode categorical variables
        merged_data['feeding_habit'] = pd.Categorical(merged_data['feeding_habit'])
        merged_data['habitat'] = pd.Categorical(merged_data['habitat'])
        merged_data['feeding_habit_code'] = merged_data['feeding_habit'].cat.codes
        merged_data['habitat_code'] = merged_data['habitat'].cat.codes
        
        # Split the data into features and target variable
        self.X = merged_data[['size', 'shape', 'composition', 'concentration', 'feeding_habit_code', 'habitat_code']]
        self.y = merged_data['ingestion_risk']
        
    def train_risk_model(self):
        """
        Train the microplastic ingestion risk assessment model using Random Forest Classifier.
        
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
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_model.fit(X_train, y_train)
        
        # Evaluate the model performance
        y_pred = self.risk_model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    def assess_ingestion_risk(self, microplastic_data, species_data):
        """
        Assess the microplastic ingestion risk for given microplastic and species data.
        
        :param microplastic_data: DataFrame containing microplastic data for risk assessment.
        :param species_data: DataFrame containing species data for risk assessment.
        
        :return: DataFrame with ingestion risk assessment results.
        
        Possible errors:
        - KeyError: If the required columns are missing in the input data.
        - ValueError: If the input data contains invalid or missing values.
        
        Solutions:
        - Ensure the input data has the same format and columns as the training data.
        - Handle missing values appropriately (e.g., remove rows or fill with default values).
        - Validate the data types and ranges of the input data.
        """
        # Validate input data
        required_columns = ['size', 'shape', 'composition', 'concentration']
        if not all(column in microplastic_data.columns for column in required_columns):
            raise KeyError(f"Microplastic data is missing required columns. Expected: {required_columns}")
        
        required_columns = ['species', 'feeding_habit', 'habitat']
        if not all(column in species_data.columns for column in required_columns):
            raise KeyError(f"Species data is missing required columns. Expected: {required_columns}")
        
        # Preprocess input data
        merged_data = pd.merge(microplastic_data, species_data, on=['size', 'shape', 'composition'])
        merged_data['feeding_habit'] = pd.Categorical(merged_data['feeding_habit'])
        merged_data['habitat'] = pd.Categorical(merged_data['habitat'])
        merged_data['feeding_habit_code'] = merged_data['feeding_habit'].cat.codes
        merged_data['habitat_code'] = merged_data['habitat'].cat.codes
        
        # Prepare input features
        X_input = merged_data[['size', 'shape', 'composition', 'concentration', 'feeding_habit_code', 'habitat_code']]
        
        # Assess ingestion risk using the trained model
        ingestion_risk = self.risk_model.predict(X_input)
        
        # Create a DataFrame with risk assessment results
        risk_assessment_results = pd.DataFrame({
            'species': merged_data['species'],
            'microplastic_id': merged_data['id'],
            'ingestion_risk': ingestion_risk
        })
        
        return risk_assessment_results
    
    def visualize_risk_assessment(self, risk_assessment_results):
        """
        Visualize the microplastic ingestion risk assessment results.
        
        :param risk_assessment_results: DataFrame containing risk assessment results.
        
        Possible errors:
        - ImportError: If the required visualization libraries are not installed.
        - ValueError: If the input data is empty or has an inconsistent format.
        
        Solutions:
        - Install the required visualization libraries (e.g., matplotlib, seaborn).
        - Ensure the input data is not empty and has the expected columns.
        """
        # Import visualization libraries
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Validate input data
        if risk_assessment_results.empty:
            raise ValueError("Risk assessment results are empty.")
        
        # Count the number of species at each risk level
        risk_counts = risk_assessment_results['ingestion_risk'].value_counts()
        
        # Create a bar plot of risk levels
        plt.figure(figsize=(8, 6))
        sns.barplot(x=risk_counts.index, y=risk_counts.values)
        plt.xlabel('Ingestion Risk Level')
        plt.ylabel('Number of Species')
        plt.title('Microplastic Ingestion Risk Assessment')
        plt.show()
        
        # Create a heatmap of risk levels by species and microplastic type
        heatmap_data = risk_assessment_results.pivot_table(index='species', columns='microplastic_id', values='ingestion_risk', aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f')
        plt.xlabel('Microplastic ID')
        plt.ylabel('Species')
        plt.title('Ingestion Risk Heatmap')
        plt.show()

def main():
    # Load microplastic and species data from files or databases
    microplastic_data = pd.read_csv('microplastic_data.csv')
    species_data = pd.read_csv('species_data.csv')
    
    # Create an instance of the risk assessment tool
    risk_assessment_tool = MicroplasticIngestionRiskAssessment(microplastic_data, species_data)
    
    # Preprocess the data
    risk_assessment_tool.preprocess_data()
    
    # Train the risk assessment model
    risk_assessment_tool.train_risk_model()
    
    # Assess ingestion risk for new microplastic and species data
    new_microplastic_data = pd.read_csv('new_microplastic_data.csv')
    new_species_data = pd.read_csv('new_species_data.csv')
    risk_assessment_results = risk_assessment_tool.assess_ingestion_risk(new_microplastic_data, new_species_data)
    
    # Visualize the risk assessment results
    risk_assessment_tool.visualize_risk_assessment(risk_assessment_results)

if __name__ == '__main__':
    main()
