import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class MicroplasticImageAnalysis:
    def __init__(self, model_path):
        """
        Initialize the MicroplasticImageAnalysis module.

        :param model_path: Path to the pre-trained deep learning model.
        """
        self.model = self.load_model(model_path)
        self.class_labels = ['Fragment', 'Fiber', 'Sphere', 'Film', 'Foam']

    def load_model(self, model_path):
        """
        Load the pre-trained deep learning model.

        :param model_path: Path to the model file.
        :return: Loaded model.

        Possible errors:
        - FileNotFoundError: If the model file is not found.
        - ValueError: If the model file is not compatible or corrupted.

        Solutions:
        - Ensure the model file exists at the specified path.
        - Ensure the model file is compatible with the current TensorFlow version.
        """
        try:
            model = load_model(model_path)
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return None
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            return None

    def preprocess_image(self, image_path):
        """
        Preprocess the input image for analysis.

        :param image_path: Path to the input image.
        :return: Preprocessed image array.

        Possible errors:
        - FileNotFoundError: If the image file is not found.
        - PIL.UnidentifiedImageError: If the image file is corrupted or not supported.

        Solutions:
        - Ensure the image file exists at the specified path.
        - Ensure the image file is in a supported format (e.g., JPEG, PNG).
        """
        try:
            image = Image.open(image_path)
            image = image.resize((224, 224))  # Resize to match the model input size
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array /= 255.0  # Normalize pixel values
            return image_array
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Image.UnidentifiedImageError:
            print(f"Error: Unable to open image file at {image_path}")
            return None

    def predict_microplastic_type(self, image_array):
        """
        Predict the type of microplastic in the image.

        :param image_array: Preprocessed image array.
        :return: Predicted microplastic type and confidence score.

        Possible errors:
        - ValueError: If the input image array is not valid.

        Solutions:
        - Ensure the input image array is properly preprocessed and has the expected shape.
        """
        try:
            predictions = self.model.predict(image_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_label = self.class_labels[predicted_class_index]
            confidence_score = predictions[0][predicted_class_index]
            return predicted_class_label, confidence_score
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            return None, None

    def analyze_image(self, image_path):
        """
        Analyze the microplastic image and return the predicted type and confidence score.

        :param image_path: Path to the input image.
        :return: Predicted microplastic type and confidence score.
        """
        image_array = self.preprocess_image(image_path)
        if image_array is not None:
            predicted_type, confidence_score = self.predict_microplastic_type(image_array)
            return predicted_type, confidence_score
        else:
            return None, None

    def analyze_image_directory(self, directory_path):
        """
        Analyze all images in a directory and return the results.

        :param directory_path: Path to the directory containing the images.
        :return: List of analysis results for each image.

        Possible errors:
        - FileNotFoundError: If the directory is not found.
        - PermissionError: If there is insufficient permission to access the directory.

        Solutions:
        - Ensure the directory exists at the specified path.
        - Ensure the script has sufficient permissions to read the directory.
        """
        try:
            image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
            results = []
            for image_file in image_files:
                image_path = os.path.join(directory_path, image_file)
                predicted_type, confidence_score = self.analyze_image(image_path)
                if predicted_type is not None:
                    results.append({'image': image_file, 'type': predicted_type, 'confidence': confidence_score})
            return results
        except FileNotFoundError:
            print(f"Error: Directory not found at {directory_path}")
            return []
        except PermissionError:
            print(f"Error: Insufficient permission to access directory at {directory_path}")
            return []

    def save_analysis_results(self, results, output_file):
        """
        Save the analysis results to a file.

        :param results: List of analysis results.
        :param output_file: Path to the output file.

        Possible errors:
        - PermissionError: If there is insufficient permission to write the file.
        - IOError: If there is an error writing the file.

        Solutions:
        - Ensure the script has sufficient permissions to write the file.
        - Ensure there is enough disk space available.
        """
        try:
            with open(output_file, 'w') as file:
                file.write("Image,Type,Confidence\n")
                for result in results:
                    file.write(f"{result['image']},{result['type']},{result['confidence']}\n")
            print(f"Analysis results saved to {output_file}")
        except PermissionError:
            print(f"Error: Insufficient permission to write file at {output_file}")
        except IOError as e:
            print(f"Error: {str(e)}")

    def visualize_results(self, results, output_file):
        """
        Visualize the analysis results using matplotlib.

        :param results: List of analysis results.
        :param output_file: Path to save the visualization image.

        Possible errors:
        - ImportError: If matplotlib is not installed.
        - PermissionError: If there is insufficient permission to write the image file.

        Solutions:
        - Ensure matplotlib is installed (`pip install matplotlib`).
        - Ensure the script has sufficient permissions to write the image file.
        """
        try:
            import matplotlib.pyplot as plt

            type_counts = {}
            for result in results:
                microplastic_type = result['type']
                if microplastic_type in type_counts:
                    type_counts[microplastic_type] += 1
                else:
                    type_counts[microplastic_type] = 1

            types = list(type_counts.keys())
            counts = list(type_counts.values())

            plt.figure(figsize=(8, 6))
            plt.bar(types, counts)
            plt.xlabel('Microplastic Type')
            plt.ylabel('Count')
            plt.title('Microplastic Type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        except ImportError:
            print("Error: Matplotlib not found. Please install matplotlib to visualize the results.")
        except PermissionError:
            print(f"Error: Insufficient permission to write image file at {output_file}")

def main():
    model_path = 'path/to/pretrained/model.h5'
    image_directory = 'path/to/image/directory'
    output_file = 'analysis_results.csv'
    visualization_file = 'analysis_visualization.png'

    analysis = MicroplasticImageAnalysis(model_path)
    results = analysis.analyze_image_directory(image_directory)
    analysis.save_analysis_results(results, output_file)
    analysis.visualize_results(results, visualization_file)

if __name__ == '__main__':
    main()
