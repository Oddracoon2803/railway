import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
import joblib

# =======================
# CONFIG
# =======================
CONF_THRESHOLD = 0.3

# ambil port dari Railway
PORT = int(os.environ.get("PORT", 8000))

# base directory project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =======================
# INIT APP
# =======================
app = Flask(__name__)
CORS(app)

# =======================
# LOAD MODEL
# =======================
print("🔄 Loading model...")

model = keras.models.load_model(
    os.path.join(BASE_DIR, "bisindo_model.h5"),
    compile=False
)

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

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

        return jsonify({
            "label": label,
            "conf": conf
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({
            "label": "-",
            "conf": 0.0,
            "error": str(e)
        })

# =======================
# LOCAL RUN (optional)
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
