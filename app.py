import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================
model = joblib.load("model_dropout.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# =========================
# FEATURES ATTENDUES
# =========================
FEATURES = [
    "age",
    "gender",
    "average_grade",
    "absenteeism_rate",
    "internet_access",
    "study_time_hours",
    "extra_activities"
]

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validation
        for col in FEATURES:
            if col not in data:
                return jsonify({"error": f"Champ manquant: {col}"}), 400

        # Construction DataFrame avec conversion absentéisme
        input_df = pd.DataFrame([{
            "age": float(data["age"]),
            "gender": data["gender"],
            "average_grade": float(data["average_grade"]),
            "absenteeism_rate": float(data["absenteeism_rate"]) / 100,
            "internet_access": data["internet_access"],
            "study_time_hours": float(data["study_time_hours"]),
            "extra_activities": data["extra_activities"]
        }])

        # Preprocessing
        X_processed = preprocessor.transform(input_df)

        # Prédiction
        prediction = model.predict(X_processed)[0]

        # Probabilités et confiance
        if hasattr(model, "predict_proba"):
            proba_complete = model.predict_proba(X_processed)[0]
            confidence = proba_complete[prediction]  # Confiance = proba de la classe prédite
        else:
            confidence = None

        result = "À Risque Élevé" if prediction == 1 else "Faible Risque"

        # Logging
        with open("logs_predictions.csv", "a") as f:
            f.write(
                f"{data['age']},{data['gender']},{data['average_grade']},"
                f"{data['absenteeism_rate']},{data['internet_access']},"
                f"{data['study_time_hours']},{data['extra_activities']},"
                f"{prediction},{confidence}\n"
            )

        # Réponse
        return jsonify({
            "prediction": result,
            "risk_level": int(prediction),
            "confidence": float(confidence) if confidence is not None else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)