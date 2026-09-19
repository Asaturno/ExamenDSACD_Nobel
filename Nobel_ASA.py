
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.write(''' # Predicción de categoría de Premio Nobel ''')
st.image("Nobel.jpg", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

st.header('Predice la categoría del premio dependiendo de la leyenda')

# Entrada
texto = st.text_area("Introduce el texto a evaluar")

user_input_data = {'Text': texto}

nobel =  pd.read_csv('Nobel.csv', encoding='latin-1')
X = nobel["Motivation_clean"]
y = nobel["Category"]

vect = TfidfVectorizer()
X_dtm = vect.fit_transform(X)

nb = model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial'
)
nb.fit(X_dtm, y)

if st.button("Predecir"):

    if texto.strip() == "":
        st.warning("Introduce un texto.")
    else:

        texto_vect = vectorizer.transform([texto])

        prediction = model.predict(texto_vect)[0]

        st.subheader("Resultado")

        st.success(prediction.capitalize())
