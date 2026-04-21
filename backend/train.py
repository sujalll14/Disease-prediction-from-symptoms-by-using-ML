"""
train_model.py
─────────────────────────────────────────────────────────────────
Disease Prediction – Dataset Generation & Model Training Script
Run: python backend/train_model.py
─────────────────────────────────────────────────────────────────
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Disease → Symptom Mapping ──────────────────────────────────────────────────
disease_symptoms = {
    "Flu":           ["fever","cough","fatigue","body_pain","headache","chills","sore_throat","runny_nose"],
    "Common Cold":   ["cough","runny_nose","sneezing","sore_throat","mild_fever","congestion"],
    "COVID-19":      ["fever","cough","fatigue","loss_of_smell","loss_of_taste","shortness_of_breath","headache","body_pain"],
    "Typhoid":       ["fever","headache","abdominal_pain","vomiting","diarrhea","fatigue","loss_of_appetite"],
    "Malaria":       ["fever","chills","sweating","headache","vomiting","fatigue","muscle_pain"],
    "Dengue":        ["fever","severe_headache","joint_pain","rash","vomiting","fatigue","eye_pain"],
    "Pneumonia":     ["cough","fever","shortness_of_breath","chest_pain","fatigue","chills","sweating"],
    "Tuberculosis":  ["cough","fever","night_sweats","weight_loss","fatigue","chest_pain","coughing_blood"],
    "Diabetes":      ["frequent_urination","excessive_thirst","fatigue","blurred_vision","slow_healing","weight_loss"],
    "Hypertension":  ["headache","dizziness","blurred_vision","chest_pain","shortness_of_breath","nausea"],
    "Asthma":        ["shortness_of_breath","wheezing","cough","chest_tightness","fatigue"],
    "Gastritis":     ["abdominal_pain","nausea","vomiting","bloating","loss_of_appetite","indigestion"],
    "Migraine":      ["severe_headache","nausea","vomiting","light_sensitivity","sound_sensitivity","blurred_vision"],
    "Chickenpox":    ["rash","fever","itching","fatigue","headache","loss_of_appetite","blister"],
    "Measles":       ["rash","fever","cough","runny_nose","red_eyes","sore_throat","blister"],
    "Jaundice":      ["yellowing_skin","dark_urine","fatigue","abdominal_pain","nausea","fever","loss_of_appetite"],
    "Appendicitis":  ["abdominal_pain","nausea","vomiting","fever","loss_of_appetite","bloating"],
    "Kidney_Stone":  ["severe_back_pain","abdominal_pain","nausea","vomiting","frequent_urination","blood_in_urine"],
    "Arthritis":     ["joint_pain","swelling","stiffness","fatigue","reduced_motion","warmth_in_joints"],
    "Anemia":        ["fatigue","pale_skin","shortness_of_breath","dizziness","headache","cold_hands"],
}

all_symptoms = sorted(set(s for syms in disease_symptoms.values() for s in syms))

print("=" * 60)
print("  Disease Prediction System — Model Trainer")
print("=" * 60)
print(f"  Total diseases  : {len(disease_symptoms)}")
print(f"  Total symptoms  : {len(all_symptoms)}")

# ── Generate Synthetic Dataset ─────────────────────────────────────────────────
print("\n[1/5] Generating dataset ...")
np.random.seed(42)
rows = []
for disease, symptoms in disease_symptoms.items():
    for _ in range(200):
        row = {s: 0 for s in all_symptoms}
        for s in symptoms:
            row[s] = 1
        for s in symptoms:
            if np.random.random() < 0.20:
                row[s] = 0
        for s in all_symptoms:
            if s not in symptoms and np.random.random() < 0.05:
                row[s] = 1
        row["disease"] = disease
        rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(DATASET_DIR, "disease_dataset.csv")
df.to_csv(csv_path, index=False)
print(f"  Dataset shape   : {df.shape}")
print(f"  Saved to        : {csv_path}")

# ── Pre-processing ─────────────────────────────────────────────────────────────
print("\n[2/5] Preprocessing ...")
X = df[all_symptoms]
y = df["disease"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"  Train samples   : {X_train.shape[0]}")
print(f"  Test  samples   : {X_test.shape[0]}")

# ── Train & Compare Models ─────────────────────────────────────────────────────
print("\n[3/5] Training models ...\n")
models = {
    "Decision Tree" : DecisionTreeClassifier(random_state=42),
    "Random Forest" : RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "Naive Bayes"   : GaussianNB(),
    "KNN"           : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM"           : SVC(kernel="rbf", probability=True, random_state=42),
}

results        = {}
trained_models = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    results[name]        = round(acc * 100, 2)
    trained_models[name] = mdl
    bar = "█" * int(acc * 30)
    print(f"  {name:<18} {bar:<30}  {acc*100:.2f}%")

best_name  = max(results, key=results.get)
best_acc   = results[best_name]
best_model = trained_models[best_name]

print(f"\n  Best Model : {best_name}  ({best_acc}%)")

y_pred_best = best_model.predict(X_test)
print("\n[4/5] Classification Report (Best Model):\n")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# ── Save Artifacts ─────────────────────────────────────────────────────────────
print("[5/5] Saving model artifacts ...")

with open(os.path.join(MODELS_DIR, "disease_model.pkl"),  "wb") as f: pickle.dump(best_model, f)
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"),  "wb") as f: pickle.dump(le, f)
with open(os.path.join(MODELS_DIR, "symptoms_list.json"), "w")  as f: json.dump(all_symptoms, f)

model_info = {
    "best_model"     : best_name,
    "accuracy"       : best_acc,
    "all_results"    : results,
    "diseases"       : list(disease_symptoms.keys()),
    "total_symptoms" : len(all_symptoms),
}
with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

print(f"  Saved 4 files to: {MODELS_DIR}")
print("\n" + "=" * 60)
print("  Training complete! Now run:  python backend/app.py")
print("=" * 60)
