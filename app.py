from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
from flask_cors import CORS



app = Flask(__name__)
model = YOLO('runs/detect/train2/weights/best.pt')
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_file = request.files['image']
    img_bytes = img_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)[0]  # Predict
    annotated = results.plot()  # Vẽ kết quả nhận dạng lên ảnh

    # Lưu ảnh tạm và trả về
    output_path = f"temp/{uuid.uuid4().hex}.jpg"
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
