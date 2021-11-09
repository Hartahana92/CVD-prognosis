import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import metrics
st.title("РЕЗУЛЬТАТЫ МЕТАБОЛОМНОГО ПРОФИЛИРОВАНИЯ")
st.write("Вероятность развития ССЗ")
data = st.file_uploader("Загрузите файл")
if data is not None:
    data = pd.read_excel(data)
    data_aa = data[['Глицин','Аланин','Пролин','Валин','Лейцин','Изолейцин','Орнитин','Аспаргиновая кислота',	'Фенилаланин','Аргинин','Цитруллин','Серин','Треонин','Лизин','Тирозин','Метионин']]
    st.write('Информация о пациенте:')
    st.write('ФИО', data.iat[0,0])
    st.write('Дата рождения', data.iat[0,1])
    st.write('Пол', data.iat[0,2])
    st.write('№ анализа', data.iat[0,3])
    st.write('Объект', data.iat[0,4])
    loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    st.write(data_aa.T)
    scaler_sv = pickle.load(open('scaler_sv.pkl', 'rb'))
    data = scaler_sv.transform(data)
    data = preprocessing.normalize(data, norm='l2')
    a=loaded_model.predict_proba(data)
    b=float(a[:,1])*100
    st.write('вероятность CCЗ = ', round(b,2), '%')
    if b>=75:
        loaded_model2 = pickle.load(open('finalized_model_2.pkl', 'rb'))
        c=loaded_model2.predict_proba(data)
        d=float(c[:,1])*100
        st.write('вероятность развития ГБ = ', round(d,2), '%')