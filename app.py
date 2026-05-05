import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
import joblib
import threading
from pyngrok import ngrok

# =======================
# CONFIG
# =======================
CONF_THRESHOLD = 0.3
PORT = 5000

# =======================
# INIT APP
# =======================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =======================
# LOAD MODEL
# =======================
model = keras.models.load_model("/content/bisindo_model.h5", compile=False)
scaler = joblib.load("/content/scaler.pkl")
label_encoder = joblib.load("/content/label_encoder.pkl")

labels = label_encoder.classes_

print("✅ Model ready")
print("Classes:", labels)

# =======================
# PREPROCESS
# =======================
def preprocess_landmarks(data):
    X = np.array(data, dtype=np.float32)

    if len(X) != 126:
        raise ValueError(f"Invalid length: {len(X)}")

    X = X.reshape(1, 126)
    X = scaler.transform(X)

    return X

# =======================
# ROUTES
# =======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "classes": len(labels)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json(silent=True) or {}
        data = json_data.get("landmarks", [])

        if not data or len(data) != 126:
            return jsonify({"label": "-", "conf": 0.0})

        X = preprocess_landmarks(data)

        pred = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        label = str(labels[idx])

        if conf < CONF_THRESHOLD:
            return jsonify({"label": "-", "conf": conf})

        return jsonify({"label": label, "conf": conf})

    except Exception as e:
        return jsonify({"error": str(e)})

# =======================
# RUN FLASK (THREAD)
# =======================
def run_app():
    app.run(host="0.0.0.0", port=PORT)

threading.Thread(target=run_app).start()
