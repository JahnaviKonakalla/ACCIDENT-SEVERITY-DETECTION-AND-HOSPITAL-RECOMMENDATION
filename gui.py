from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Load model and define class names
model = tf.keras.models.load_model("accident_prediction_model.h5")
class_names = ['major accident', 'mini accident']

# Load and preprocess image
image_path = "C:\\Users\\manik\\OneDrive\\Desktop\\Rk\\jashuva\\Road acd\\Dataset\\mini accident\\2.jpg"  # Use a known image path
img = load_img(image_path, target_size=(64, 64))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = predictions[0][predicted_class]

print(f"Predicted Class: {class_names[predicted_class]}")
print(f"Confidence: {confidence}")
