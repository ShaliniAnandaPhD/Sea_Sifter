import pandas as pd

def preprocess_data(species_file, env_file, climate_file):
    species_data = pd.read_csv(species_file)
    env_data = pd.read_csv(env_file)
    climate_data = pd.read_csv(climate_file)
    
    # Perform data preprocessing steps
    preprocessed_data = pd.merge(species_data, env_data, on=['location', 'date'])
    preprocessed_data = pd.merge(preprocessed_data, climate_data, on='date')
    preprocessed_data = preprocessed_data.fillna(method='ffill')
    preprocessed_data['temperature_anomaly'] = preprocessed_data['temperature'] - preprocessed_data['temperature'].mean()
    
    return preprocessed_data.to_dict(orient='records')
