"""
app.py  -  Flask Backend API
─────────────────────────────────────────────────────────────────
Disease Prediction System
Run: python backend/app.py
Open: http://127.0.0.1:5000
─────────────────────────────────────────────────────────────────
"""

import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)

# ── Load Model Artifacts ───────────────────────────────────────────────────────
def load_artifacts():
    missing = []
    for fname in ["disease_model.pkl", "label_encoder.pkl", "symptoms_list.json", "model_info.json"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"\nMissing model files: {missing}\n"
            f"Please run first:  python backend/train_model.py\n"
        )
    with open(os.path.join(MODELS_DIR, "disease_model.pkl"),  "rb") as f: model         = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"),  "rb") as f: le            = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "symptoms_list.json"), "r")  as f: symptoms_list = json.load(f)
    with open(os.path.join(MODELS_DIR, "model_info.json"),    "r")  as f: model_info    = json.load(f)
    return model, le, symptoms_list, model_info

model, le, symptoms_list, model_info = load_artifacts()

# ── Disease Advice ─────────────────────────────────────────────────────────────
disease_advice = {
    "Flu":          "Rest, stay hydrated. Take paracetamol for fever. Seek medical care if symptoms worsen.",
    "Common Cold":  "Rest and drink fluids. OTC cold medicine may help. Usually resolves in 7-10 days.",
    "COVID-19":     "Isolate immediately. Monitor oxygen levels. Contact a doctor if breathing is difficult.",
    "Typhoid":      "See a doctor promptly - antibiotics are needed. Drink only purified water.",
    "Malaria":      "Urgent medical care required. Anti-malarial medication must be prescribed by a doctor.",
    "Dengue":       "No specific antiviral. Stay hydrated, avoid aspirin/ibuprofen. Hospital if severe.",
    "Pneumonia":    "Medical attention needed. Antibiotics (if bacterial). Rest and plenty of fluids.",
    "Tuberculosis": "Requires long-term antibiotic treatment. Consult a pulmonologist immediately.",
    "Diabetes":     "Monitor blood sugar regularly. Consult an endocrinologist for medication/diet plan.",
    "Hypertension": "Reduce salt, exercise, manage stress. Consult a cardiologist for medication.",
    "Asthma":       "Use prescribed inhaler. Avoid triggers. Carry rescue inhaler at all times.",
    "Gastritis":    "Avoid spicy/acidic foods and alcohol. Antacids may help. See a doctor if persistent.",
    "Migraine":     "Rest in a dark, quiet room. Pain relievers may help. Consult a neurologist.",
    "Chickenpox":   "Rest and avoid scratching. Calamine lotion for itching. Isolate to prevent spread.",
    "Measles":      "Rest and fluids. Vitamin A supplements may help. Consult a doctor immediately.",
    "Jaundice":     "Urgent medical evaluation needed. Avoid alcohol. Eat light, easily digestible food.",
    "Appendicitis": "EMERGENCY - go to hospital immediately. Surgery (appendectomy) is usually required.",
    "Kidney_Stone": "Drink plenty of water. Pain management. See a urologist - some stones need treatment.",
    "Arthritis":    "Anti-inflammatory medication, physical therapy. Consult a rheumatologist.",
    "Anemia":       "Iron-rich diet (leafy greens, meat). Iron supplements if prescribed. Blood test needed.",
}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info")
def api_info():
    return jsonify({"symptoms": symptoms_list, "model_info": model_info})

@app.route("/api/diseases")
def api_diseases():
    return jsonify({"diseases": model_info.get("diseases", [])})

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json(force=True)
    vec      = np.array([[int(data.get(s, 0)) for s in symptoms_list]])
    pred_idx = int(model.predict(vec)[0])
    pred_name = le.inverse_transform([pred_idx])[0]
    advice   = disease_advice.get(pred_name, "Consult a qualified medical professional.")

    top_predictions = []
    confidence      = 0.85

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        confidence = float(proba[pred_idx])
        top_idx    = np.argsort(proba)[::-1][:6]
        top_predictions = [
            {"disease": le.inverse_transform([int(i)])[0], "probability": round(float(proba[i]), 4)}
            for i in top_idx
        ]

    return jsonify({
        "predicted_disease" : pred_name,
        "confidence"        : round(confidence, 4),
        "advice"            : advice,
        "model_used"        : f"{model_info['best_model']} ({model_info['accuracy']}% accuracy)",
        "top_predictions"   : top_predictions,
    })

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Disease Prediction System - Flask Backend")
    print("=" * 55)
    print(f"  Model    : {model_info['best_model']}")
    print(f"  Accuracy : {model_info['accuracy']}%")
    print(f"  Diseases : {len(model_info['diseases'])}")
    print(f"  Symptoms : {model_info['total_symptoms']}")
    print("=" * 55)
    print("  Open browser  -->  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
