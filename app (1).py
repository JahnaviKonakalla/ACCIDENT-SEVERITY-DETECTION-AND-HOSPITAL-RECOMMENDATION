import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Define the class names for the prediction
class_names = ['major accident', 'mini accident']

# Load the model dynamically to avoid session issues
def load_model():
    return tf.keras.models.load_model("accident_prediction_model.h5")

# Image preprocessing function
def preprocess_image(image_file, img_height=64, img_width=64):
    """Load and preprocess the image."""
    img = Image.open(BytesIO(image_file.read()))  # Open image from FileStorage
    img = img.resize((img_height, img_width))  # Resize the image
    img_array = np.array(img)  # Convert to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    """Render the main page with the image upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction logic when an image is uploaded."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        # Preprocess the image
        img_array = preprocess_image(image_file)

        # Load the model inside the route to avoid session issues
        model = load_model()

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])

        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Log any exceptions
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
