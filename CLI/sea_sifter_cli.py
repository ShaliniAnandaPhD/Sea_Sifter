import argparse
from data_processing import preprocess_data
from modeling import train_model
from visualization import visualize_predictions

def main():
    """
    Main function for the Sea Sifter CLI tool.
    """
    parser = argparse.ArgumentParser(description='Sea Sifter CLI Tool')
    parser.add_argument('command', choices=['preprocess', 'train', 'visualize'], help='Command to run')
    parser.add_argument('--species-file', type=str, help='Path to the marine species CSV file')
    parser.add_argument('--env-file', type=str, help='Path to the environmental factors CSV file')
    parser.add_argument('--climate-file', type=str, help='Path to the climate data CSV file')
    parser.add_argument('--output-file', type=str, help='Path to save the processed data')
    parser.add_argument('--model-file', type=str, help='Path to save the trained model')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocessed_data = preprocess_data(args.species_file, args.env_file, args.climate_file)
        preprocessed_data.to_csv(args.output_file, index=False)
        print(f"Preprocessed data saved to: {args.output_file}")
    
    elif args.command == 'train':
        preprocessed_data = pd.read_csv(args.output_file)
        model = train_model(preprocessed_data)
        joblib.dump(model, args.model_file)
        print(f"Trained model saved to: {args.model_file}")
    
    elif args.command == 'visualize':
        preprocessed_data = pd.read_csv(args.output_file)
        model = joblib.load(args.model_file)
        visualize_predictions(preprocessed_data, model)

if __name__ == '__main__':
    main()
