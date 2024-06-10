from flask import Flask, request, render_template_string, send_file
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model = load_model('rice_quality_model.h5')

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to serve the HTML form
@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Image</title>
        <!-- Bootstrap CSS -->
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .card-custom {
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .card-header-custom {
                font-weight: bold;
                font-size: 1.25rem;
                color: #333;
            }
            .btn-custom {
                background-color: #a3d6a7;
                border: none;
            }
            .form-control-file-custom {
                padding: 50px;
                border: 2px dashed #ccc;
                text-align: center;
                color: #aaa;
            } 
            .result-text {
                font-size: 2rem;
                font-weight: bold;
                color: #6c757d;
                text-align: center;
                margin-top: 20px;
            }
            .result-good {
                color: #28a745;
            }
            .result-bad {
                color: #dc3545;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-5">
                    <div class="card card-custom">
                        <div class="card-header text-center card-header-custom">
                            Upload Gambar <strong>Beras</strong>
                        </div>
                        <div class="card-body text-center">
                            <form action="/threshold" method="post" enctype="multipart/form-data">
                                <div class="form-group">
                                    <input type="file" class="form-control-file form-control-file-custom" name="file" accept="image/*">
                                </div>
                                <button type="submit" class="btn btn-custom">Upload</button>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-5">
                    <div class="card card-custom">
                        <div class="card-header text-center card-header-custom">
                            Upload Gambar Hasil <strong>Thresdebuging</strong>
                        </div>
                        <div class="card-body text-center">
                            <form action="/predict" method="post" enctype="multipart/form-data">
                                <div class="form-group">
                                    <input type="file" class="form-control-file form-control-file-custom" name="file" accept="image/*">
                                </div>
                                <button type="submit" class="btn btn-custom">Upload</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
            {% if result %}
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="result-text {% if result == 'Good' %}result-good{% else %}result-bad{% endif %}">
                        Kualitas Beras: {{ result }}
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

# Function to predict quality based on image size
def predict_quality(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Save threshold image for debugging
    thresh_path = os.path.join(UPLOAD_FOLDER, 'thresh_debug.jpg')
    cv2.imwrite(thresh_path, thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 'Bad'

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Print dimensions for debugging
    print(f"Bounding Box Dimensions: w={w}, h={h}")
    
    # Determine quality based on the bounding box dimensions
    if w > 50 and h > 50:  # Adjust these thresholds as needed
        return 'Good'
    else:
        return 'Bad'

# Route to handle file upload and thresholding
@app.route('/threshold', methods=['POST'])
def threshold():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Save threshold image
    thresh_path = os.path.join(UPLOAD_FOLDER, 'thresh_debug.jpg')
    cv2.imwrite(thresh_path, thresh)
    
    return send_file(thresh_path, as_attachment=True)

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    
    # Predict quality
    prediction = model.predict(img)
    result = 'Good' if prediction > 0.5 else 'Bad'
    
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hasil Prediksi</title>
        <!-- Bootstrap CSS -->
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                flex-direction: column;
            }
            .result-container {
                text-align: center;
            }
            .result-title {
                font-size: 1.5rem;
                font-weight: bold;
                color: #333;
            }
            .result-good {
                font-size: 3rem;
                font-weight: bold;
                color: #28a745;
            }
            .result-bad {
                font-size: 3rem;
                font-weight: bold;
                color: #dc3545;
            }
            .btn-custom {
                background-color: #a3d6a7;
                border: none;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="result-container">
            <div class="result-title">Kualitas Beras :</div>
            <div class="{{ 'result-good' if result == 'Good' else 'result-bad' }}">{{ result }}</div>
            <a href="/" class="btn btn-custom">Kembali</a>
        </div>
        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </body>
    </html>
    '''
    
    return render_template_string(html_content, result=result)

if __name__ == '__main__':
    app.run(debug=True)
