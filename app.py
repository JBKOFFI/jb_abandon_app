
import os

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

import joblib
import pandas as pd


# =========================
# CONFIGURATION FLASK
# =========================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:legende@localhost:5432/base_abandon"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================
# MODELE DE LA BASE DE DONNEES
# =========================

class Student(db.Model):
    __tablename__ = "log_prediction"

    id = db.Column(db.Integer, primary_key=True)

    # Informations scolaires
    ville = db.Column(db.String(100), nullable=False)
    etablissement = db.Column(db.String(200), nullable=False)
    niveau_etude = db.Column(db.String(100), nullable=False)

    # Variables utilisées par le modèle ML
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    average_grade = db.Column(db.Float, nullable=False)
    absenteeism_rate = db.Column(db.Float, nullable=False)
    internet_access = db.Column(db.String(10), nullable=False)
    study_time_hours = db.Column(db.Float, nullable=False)
    extra_activities = db.Column(db.String(10), nullable=False)

    # Résultat du modèle
    prediction = db.Column(db.Integer, nullable=False)
    


# =========================
# CREATION DE LA TABLE
# =========================

with app.app_context():
    db.create_all()


# =========================
# CHARGEMENT DES MODELES ML
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
# PAGE D'ACCUEIL
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# API DE PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # RECUPERATION DES DONNEES
        # =========================

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Aucune donnée reçue"
            }), 400


        # =========================
        # VALIDATION DES CHAMPS
        # =========================

        for col in FEATURES:

            if col not in data:
                return jsonify({
                    "error": f"Champ manquant : {col}"
                }), 400


        # =========================
        # PREPARATION DES DONNEES
        # =========================

        input_df = pd.DataFrame([{

            "age": float(data["age"]),

            "gender": data["gender"],

            "average_grade": float(
                data["average_grade"]
            ),

            # Conversion du pourcentage
            # Exemple : 25 -> 0.25
            "absenteeism_rate": float(
                data["absenteeism_rate"]
            ) / 100,

            "internet_access": data["internet_access"],

            "study_time_hours": float(
                data["study_time_hours"]
            ),

            "extra_activities": data["extra_activities"]

        }])


        # =========================
        # PREPROCESSING
        # =========================

        X_processed = preprocessor.transform(
            input_df
        )


        # =========================
        # PREDICTION
        # =========================

        prediction = int(
            model.predict(X_processed)[0]
        )


        # =========================
        # CALCUL DE LA CONFIANCE
        # =========================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                X_processed
            )[0]

            confidence = float(
                probabilities[prediction]
            )

        else:

            confidence = None


        # =========================
        # INTERPRETATION
        # =========================

        if prediction == 1:

            result = "À Risque Élevé"

        else:

            result = "Faible Risque"


        # =========================
        # ENREGISTREMENT POSTGRESQL
        # =========================

        student = Student(

            ville=data["ville"],

         etablissement=data["etablissement"],

         niveau_etude=data["niveau_etude"],

         age=int(data["age"]),

         gender=data["gender"],

         average_grade=float(
        data["average_grade"]
    ),

         absenteeism_rate=float(
        data["absenteeism_rate"]
    ),

        internet_access=data["internet_access"],

         study_time_hours=float(
        data["study_time_hours"]
    ),

        extra_activities=data["extra_activities"],

        prediction=prediction)


        db.session.add(student)

        db.session.commit()


        # =========================
        # REPONSE JSON
        # =========================

        return jsonify({

            "prediction": result,

            "risk_level": prediction,

            "confidence": confidence

        }), 200


    # =========================
    # GESTION DES ERREURS
    # =========================

    except Exception as e:

        # Annule la transaction PostgreSQL
        db.session.rollback()

        return jsonify({

            "error": str(e)

        }), 500


# =========================
# LANCEMENT DE L'APPLICATION
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
