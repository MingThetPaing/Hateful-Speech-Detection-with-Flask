from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
try:
    model = joblib.load('hate_speech_model.joblib')
except:
    print("Model file not found. Please train the model first.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        prediction_raw = model.predict([text])[0]
        prediction = int(prediction_raw)
        
        result = {
            'text': text,
            'prediction': prediction,
            'classification': 'non-hateful' if prediction == 0 else 'hateful'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)