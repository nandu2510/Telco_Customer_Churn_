# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv("first_telc.csv")

model = pickle.load(open("model.sav", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():

    inputs = [request.form.get(f'query{i}', "") for i in range(1, 20)]

    new_df = pd.DataFrame([inputs], columns=[
        'SeniorCitizen','MonthlyCharges','TotalCharges',
        'gender','Partner','Dependents','PhoneService',
        'MultipleLines','InternetService','OnlineSecurity',
        'OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract',
        'PaperlessBilling','PaymentMethod','tenure'
    ])

    df = pd.concat([df_1, new_df], ignore_index=True)

    # convert tenure safely
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    labels = ["{0} - {1}".format(i, i+11) for i in range(1, 72, 12)]

    df['tenure_group'] = pd.cut(
        df['tenure'],
        range(1, 80, 12),
        right=False,
        labels=labels
    )

    df.drop(columns=['tenure'], inplace=True, errors='ignore')

    # one-hot encoding
    df_dummies = pd.get_dummies(df)

    # 🔥 IMPORTANT FIX
    df_dummies = df_dummies.reindex(columns=model_columns, fill_value=0)

    X = df_dummies.tail(1)

    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if prediction == 1:
        result = "Customer WILL CHURN"
    else:
        result = "Customer WILL NOT CHURN"

    return render_template(
        "home.html",
        output1=result,
        output2=f"Confidence: {prob*100:.2f}%"
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)