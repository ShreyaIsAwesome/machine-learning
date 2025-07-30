from flask import Flask, request, jsonify, render_template
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

from model import Model
model = Model(load_existing_model=True)

def preprocess(file_data):
    if file_data.max() <= 1.0:
        file_data = (file_data * 255).astype(np.uint8)
    image = Image.fromarray(file_data)
    image = image.convert("RGB")
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 150, 150, 3)

descriptions = {
    "tumor": "This is the label for the cancerous cells themselves. In colorectal cancer, these cells often form abnormal, disorganized gland-like structures. Under the microscope, they tend to have large, dark nuclei and grow in a chaotic, uncontrolled way compared to healthy tissue.",
    "stroma": "Think of stroma as the 'support structure' or scaffolding for a tumor. It's not made of cancer cells, but rather a mix of connective tissue, blood vessels, and immune cells that the tumor recruits to help it grow and get nutrients. Analyzing the stroma is critical because it can reveal how the tumor is interacting with its environment.",
    "complex": "This label doesn't describe a single tissue type but rather a pattern. It refers to complex and crowded arrangements of glands, which are often a key feature of cancerous tissue. Healthy glands in the colon are usually simple and organized, while tumor glands can become tangled and complex.",
    "lympho": "Short for lymphocytes, these are your body's immune cells (a type of white blood cell). When you see 'lympho' tissue, you're seeing the body fighting back against the cancer. A high concentration of these cells near a tumor can sometimes be a sign of a better prognosis.",
    "adipose": "This refers to fat tissue. In the context of colorectal cancer, adipose tissue can be found around the tumor and may influence its growth. While not directly involved in the cancer process, its presence can affect how the tumor behaves and responds to treatment.",
    "mucosa": "This is the normal, healthy inner lining of the colon. Its job is to absorb water and secrete mucus to help things move along smoothly. It has a very organized structure of straight, tube-like glands called crypts. Comparing 'tumor' images to 'mucosa' images shows a clear difference between chaos and order.",
    "debris": "This label refers to cellular debris, which is the leftover material from dead or dying cells. In cancerous tissue, you might see more debris due to the rapid growth and death of tumor cells. It's a sign that the tumor is active and changing, but it doesn't provide much information about the type of cancer itself.",
    "empty": "This label is used for patches on the slide that contain no tissue at all. It could be a hole in the tissue sample or just the edge of the glass slide where no cells are present. It's important to identify these areas so that they don't get misclassified as a type of tissue. In the context of cancer, empty patches are not informative but are part of the overall slide analysis.",
}


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    file = request.files['image']

    encoded_file = imread(file)
    print(f"Encoded file shape: {encoded_file.shape}")
    nn_image = preprocess(encoded_file)
    predicted_value = model.predict(nn_image)

    print(f"Predicted value: {predicted_value}")
    print(descriptions[predicted_value])

    return jsonify({'message': f"{predicted_value}", "description": descriptions[predicted_value]})

if __name__ == '__main__':
    url = 'https://drive.google.com/uc?id=1rPrwRTzHqDzMj_2-EcMLgHWvg61tHg_-'
    output = 'model/model.keras'
    gdown.download(url, output, quiet=False)
    app.run(debug=True)
