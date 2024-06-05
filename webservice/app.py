from flask import Flask, request, jsonify
from sea_sifter.data_processing import preprocess_data
from sea_sifter.modeling import train_model
from sea_sifter.visualization import visualize_predictions

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    preprocessed_data = preprocess_data(data['species_file'], data['env_file'], data['climate_file'])
    return jsonify(preprocessed_data)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    model = train_model(data['preprocessed_data'])
    return jsonify(model)

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    image_path = visualize_predictions(data['preprocessed_data'], data['model'])
    return jsonify({'image_path': image_path})

if __name__ == '__main__':
    app.run(debug=True)
