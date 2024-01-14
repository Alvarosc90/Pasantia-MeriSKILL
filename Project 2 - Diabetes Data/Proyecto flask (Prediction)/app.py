from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

archivo_csv = r"C:\Users\Pixelado\Desktop\Proyect 2\diabetes.csv"
df = pd.read_csv(archivo_csv)

X = df.drop('Outcome', axis=1)
Y = df['Outcome']
X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['diabetes_pedigree_function']),
        float(request.form['age'])
    ]

    # Realizar la predicción
    resultado_prediccion = model.predict([features])

    if resultado_prediccion[0] == 1:
        resultado = "El modelo predice que el paciente tiene diabetes."
    else:
        resultado = "El modelo predice que el paciente no tiene diabetes."

    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
