from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

# LOAD MODEL
model = joblib.load("svm_phq9_model.pkl")
scaler = joblib.load("scaler_phq9.pkl")

labels = [
    "Minimal Depression",
    "Mild Depression",
    "Moderate Depression",
    "Moderately Severe Depression",
    "Severe Depression"
]

app = Flask(__name__)

@app.route("/")
def home():
    return "SVM PHQ9 API Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        features = np.array(data["features"]).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)

        probabilities = model.predict_proba(scaled_features)

        predicted_class = int(prediction[0])

        confidence = float(np.max(probabilities))

        result = labels[predicted_class]

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
