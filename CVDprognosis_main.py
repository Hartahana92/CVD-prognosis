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
    data_aa = data[
        ['Глицин', 'Аланин', 'Пролин', 'Валин', 'Лейцин', 'Изолейцин', 'Орнитин', 'Аспаргиновая кислота', 'Фенилаланин',
         'Аргинин', 'Цитруллин', 'Серин', 'Треонин', 'Лизин', 'Тирозин', 'Метионин']]
    st.write('Информация о пациенте:')
    st.write('ФИО', data.iat[0, 0])
    st.write('Дата рождения', data.iat[0, 1])
    st.write('Пол', data.iat[0, 2])
    st.write('№ анализа', data.iat[0, 3])
    st.write('Объект', data.iat[0, 4])
    data0 = pd.read_excel('data0.xlsx')
    data_aa1 = data_aa.T
    data_aa1 = data_aa1.reset_index()
    data_aa1.columns = ['Метаболит', 'Результат']
    aa = data0.merge(data_aa1, how='left', left_on='Метаболит', right_on='Метаболит')
    df = aa.copy()


    def make_result_column(x):
        if x[0] < x[2] < x[1]:
            return 'Норма'
        if x[2] / 5 < x[1] < x[2]:
            return 'Риск повышения'
        if x[2] * 5 > x[0] > x[2]:
            return 'Риск понижения'
        if x[2] / 5 > x[1]:
            return 'Повышено'
        else:
            return 'Понижено'


    df['Вывод'] = df[df.columns[1:4]].apply(lambda x: make_result_column(x), axis=1)
    rel_up = df['Результат'] / df['Верхняя граница']
    df['Риск повышения'] = rel_up.loc[rel_up.between(1, 5)]
    df['Повышено'] = rel_up.loc[rel_up.between(5, np.inf)]
    df['Норма'] = rel_up.loc[rel_up.between(0, 1)]
    rel_down = df['Нижняя граница'] / df['Результат']
    df['Риск понижения'] = rel_down.loc[rel_down.between(1, 5)]
    df['Понижено'] = rel_down.loc[rel_down.between(5, np.inf)]

    df = df[['Метаболит', 'Нижняя граница', 'Верхняя граница', 'Результат', 'Вывод',
             'Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']]


    def color(x, props=''):
        if type(x) != str:
            if x > 5:
                return "background-color: lightred"
            if 5 > x > 1:
                return "background-color: lightyellow"
            if x < 1:
                return "background-color: lightgreen"
            else:
                return "background-color: None"
        else:
            return "background-color: None"


    color_cols = ['Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']
    st.dataframe(df.style.applymap(lambda x: color(x), subset=color_cols) \
                 .format(na_rep=' ', precision=2, subset=df.columns) \
                 .format(na_rep=' ', precision=0, subset=['Нижняя граница', 'Верхняя граница']))

    # loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
    # st.write(data_aa.T)
    # scaler_sv = pickle.load(open('scaler_sv.pkl', 'rb'))
    # data = scaler_sv.transform(data)
    # data = preprocessing.normalize(data, norm='l2')
    # a=loaded_model.predict_proba(data)
    # b=float(a[:,1])*100
    # st.write('вероятность CCЗ = ', round(b,2), '%')
    # if b>=75:
    #   loaded_model2 = pickle.load(open('finalized_model_2.pkl', 'rb'))
    #  c=loaded_model2.predict_proba(data)
    # d=float(c[:,1])*100
    # st.write('вероятность развития ГБ = ', round(d,2), '%')
