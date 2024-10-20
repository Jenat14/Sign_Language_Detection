from flask import Flask, request, jsonify, render_template
from keras._tf_keras.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = load_model('asl_finetuned_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
