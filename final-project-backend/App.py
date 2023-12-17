# app.py

from flask import Flask, request, jsonify, make_response, render_template
import joblib
import cv2
import io
import imutils
import numpy as np
from PIL import Image
import base64
from flask_cors import CORS  # Import the CORS class

app = Flask(__name__)
CORS(app)
LB = joblib.load('./label_binarizer.joblib')

# Load your pre-trained model
model = joblib.load("./pre-trained-models/model_trail_3.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        print(request)
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        # Make predictions using your pre-trained model
        letter,image = get_letters(file)
        word = get_word(letter)
        print(word)

        # Save the processed image to a BytesIO object
        processed_image_io = convert_image_to_png(image)

        # Return the processed image as a response
        response = make_response({'prediction': word, 'image': base64.b64encode(processed_image_io.getvalue()).decode('utf-8')})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        return jsonify({'error': str(e)})

def sort_contours(cnts, method="left-to-right"):
    reverse = method in ["right-to-left", "bottom-to-top"]
    i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def convert_image_to_png(image):
    # Convert the OpenCV image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save the PIL Image to a BytesIO object as PNG
    processed_image_io = io.BytesIO()
    pil_image.save(processed_image_io, format="PNG")
    processed_image_io.seek(0)

    return processed_image_io

def get_letters(img):
    letters = []
    image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1, 32, 32, 1)
        ypred = model.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)
    
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word


@app.route('/')
def home():
    # response = make_response()
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
