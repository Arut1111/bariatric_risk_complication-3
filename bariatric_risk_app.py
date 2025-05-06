
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model_xgb.json")
    return model

st.set_page_config(page_title="Риск осложнений", page_icon="favicon.png")

st.title("Прогноз риска послеоперационных осложнений (бариатрическая хирургия)")
st.write("Введите послеоперационные параметры пациента для оценки вероятности осложнений")

rv = st.selectbox("Была ли рвота после операции?", ("Да", "Нет"))
pain = st.slider("Абдоминальная боль (ВАШ)", 0, 10, 2)
hr = st.number_input("Частота сердечных сокращений (уд/мин)", 40, 180, 76)
bp = st.text_input("Артериальное давление (пример: 120/80)", "120/80")

try:
    systolic, diastolic = map(int, bp.split("/"))
    bp_mean = (systolic + 2 * diastolic) / 3
except:
    bp_mean = 93

rv_binary = 1 if rv == "Да" else 0

input_data = pd.DataFrame({
    "Наличие рвоты после операции": [rv_binary],
    "Абдоминальная боль, ВАШ": [pain],
    "ЧСС, уд/мин": [hr],
    "АД, мм.рт.ст": [bp_mean]
})

model = load_model()

if st.button("Рассчитать риск"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Вероятность осложнений: {prob:.2%}")
    if prob > 0.5:
        st.warning("Высокий риск осложнений. Требуется дополнительное наблюдение!")
    else:
        st.info("Риск осложнений низкий. Продолжайте стандартное наблюдение.")
