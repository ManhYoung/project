from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('vehicle_classification_model.keras')
class_names = ['Car', 'Truck', 'Motorcycle', 'Bus']

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Chuẩn hóa ảnh nếu cần
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file)
        processed_image = preprocess_image(image, target_size=(128, 128))  # Kích thước ảnh phù hợp với mô hình
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        return jsonify({"class": predicted_class, "confidence": float(confidence)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
