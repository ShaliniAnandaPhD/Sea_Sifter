import matplotlib.pyplot as plt
import seaborn as sns

def visualize_predictions(preprocessed_data, model):
    data = pd.DataFrame(preprocessed_data)
    predictions = model.predict(data.drop(['species_population'], axis=1))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['date'], y=data['species_population'], label='Actual')
    sns.lineplot(x=data['date'], y=predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Species Population')
    plt.title('Marine Species Population Predictions')
    plt.legend()
    
    image_path = 'output/visualizations/species_population_predictions.png'
    plt.savefig(image_path)
    
    return image_path
