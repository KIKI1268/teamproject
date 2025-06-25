from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('garbage_classifier.pkl')

@app.route('/')
def index():
    return open('EAI6020_FINAL.html').read()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = model.predict([data['features']])
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
