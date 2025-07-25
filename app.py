from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def predict():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        return jsonify({'message': f"{files}"})
