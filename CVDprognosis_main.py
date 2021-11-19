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
datazero = st.file_uploader("Загрузите файл")
if datazero is not None:
    data = pd.read_excel(datazero)
    data_aa = data[
        ['Глицин', 'Аланин', 'Пролин', 'Валин', 'Лейцин', 'Изолейцин', 'Орнитин', 'Аспаргиновая кислота', 'Фенилаланин',
         'Аргинин', 'Цитруллин', 'Серин', 'Треонин', 'Лизин', 'Тирозин', 'Метионин']]
    st.write('Информация о пациенте:')
    info = pd.read_excel('patient.xlsx')
    info['ФИО'] = data['ФИО']
    info['Дата рождения'] = data['Дата рождения']
    info['Пол'] = data['Пол']
    info['№ анализа'] = data['Номер']
    info['Объект'] = data['Объект']
    st.dataframe(info)
    del data['ФИО']
    del data['Дата рождения']
    del data['Пол']
    del data['Номер']
    del data['Объект']
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
                return "background-color: red"
            if 5 > x > 1:
                return "background-color: lightyellow"
            if x < 1:
                return "background-color: lightgreen"
            else:
                return "background-color: None"
        else:
            return "background-color: None"


    color_cols = ['Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']
    st.write('Аминокислоты')
    st.dataframe(df.style.applymap(lambda x: color(x), subset=color_cols) \
                 .format(na_rep=' ', precision=2, subset=df.columns) \
                 .format(na_rep=' ', precision=0, subset=['Нижняя граница', 'Верхняя граница']))
    st.write('Ацилкарнитины')
    # Карнитины
    data1 = pd.read_excel('Data1.xlsx')
    data_carn = data[['Карнитин (С0)', 'Acetylcarnitine (С2)',
                             'Propionylcarnitine (С3)', 'Butyrylcarnitine (С4)',
                             'Isovalerylcarnitine (iC5)', 'Tiglylcarnitine (C5:1)',
                             'Hexanoylcarnitine (C6)', 'Hydroxyisovalerylcarnitine (iC5-OH)',
                             'Glutarylcarnitine (С5DC)', 'Octenoylcarnitine (C8)',
                             'Octanoylcarnitine (C8)', 'Adipoylcarnitine (С6DC)',
                             'Decadienoylcarnitine (C10:2)', 'Decenoylcarnitine (C10:1)',
                             'Decanoylcarnitine (C10)', 'Dodecenoylcarnitine (C12:1)',
                             'Dodecanoylcarnitine (С12)', 'Tetradecadienoylcarnitine (C14:2)',
                             'Tetradecenoylcarnitine (C14:1)', 'Tetradecanoylcarnitine (C14)',
                             'Hydroxytetradecanoylcarnitine (C14:2-OH) ',
                             'Hexadecenoylcarnitine (C16:1) ', 'Palmitoylcarnitine (C16)',
                             'Hydroxyhexadecenoylcarnitine (C16:1-OH) ',
                             'Hydroxyhexadecanoylcarnitine (C16-OH)', 'Linoleylcarnitine (C18:2)',
                             'Oleoylcarnitine (C18:1) ', 'Stearoylcarnitine (C18)']]
    data_carn = data_carn.T
    data_carn = data_carn.reset_index()
    data_carn.columns = ['Метаболит', 'Результат']
    data_carn = data1.merge(data_carn, how='left', left_on='Метаболит', right_on='Метаболит')
    df2 = data_carn.copy()
    df2['Вывод'] = df2[df2.columns[1:4]].apply(lambda x: make_result_column(x), axis=1)
    rel_up = df2['Результат'] / df2['Верхняя граница']
    df2['Риск повышения'] = rel_up.loc[rel_up.between(1, 5)]
    df2['Повышено'] = rel_up.loc[rel_up.between(5, np.inf)]
    df2['Норма'] = rel_up.loc[rel_up.between(0, 1)]
    rel_down = df2['Нижняя граница'] / df2['Результат']
    df2['Риск понижения'] = rel_down.loc[rel_down.between(1, 5)]
    df2['Понижено'] = rel_down.loc[rel_down.between(5, np.inf)]
    df2 = df2[['Метаболит', 'Нижняя граница', 'Верхняя граница', 'Результат', 'Вывод',
               'Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']]
    color_cols = ['Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']

    st.dataframe(df2.style.applymap(lambda x: color(x), subset=color_cols) \
        .format(na_rep=' ', precision=2, subset=df2.columns) \
        .format(na_rep=' ', precision=2, subset=['Нижняя граница', 'Верхняя граница']))

    st.write('СВободный холин и СДМА')
    # Choline and SDMA
    data3 = pd.read_excel('data3.xlsx')
    data_SDMA = data[['Холин (свободный)', 'АДМА', 'СДМА', 'АДМА/Аргинин']]
    data_SDMA = data_SDMA.T
    data_SDMA = data_SDMA.reset_index()
    data_SDMA.columns = ['Метаболит', 'Результат']
    data_SDMA = data3.merge(data_SDMA, how='left', left_on='Метаболит', right_on='Метаболит')
    df3 = data_SDMA.copy()
    df3['Вывод'] = df3[df3.columns[1:4]].apply(lambda x: make_result_column(x), axis=1)
    rel_up = df3['Результат'] / df3['Верхняя граница']
    df3['Риск повышения'] = rel_up.loc[rel_up.between(1, 5)]
    df3['Повышено'] = rel_up.loc[rel_up.between(5, np.inf)]
    rel_down = df3['Нижняя граница'] / df3['Результат']
    df3['Риск понижения'] = rel_down.loc[rel_down.between(1, 5)]
    df3['Понижено'] = rel_down.loc[rel_down.between(5, np.inf)]


    def norma(x):
        if x[2] == 'Норма':
            return x[0] / x[1]


    df3['Норма'] = df3[['Нижняя граница', 'Результат', 'Вывод']].apply(lambda x: norma(x), axis=1)
    df3 = df3[['Метаболит', 'Нижняя граница', 'Верхняя граница', 'Результат', 'Вывод',
               'Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']]
    color_cols = ['Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']

    st.dataframe(df3.style.applymap(lambda x: color(x), subset=color_cols) \
        .format(na_rep=' ', precision=2, subset=df3.columns) \
        .format(na_rep=' ', precision=2, subset=['Нижняя граница', 'Верхняя граница']))
    st.write('Триптофаны')
    # триптофаны
    data4 = pd.read_excel('data4.xlsx')
    data_trp = data[['5-Hydroxytryptophan', 'Kynurenine', 'Tryptophan',
                            'Kynurenine/Tryptophan', 'Serotonin', 'Quinolinic acid', 'HIAA',
                            'Tryptamine', 'Antranillic acid', 'Xanturenic acid', 'Indole-3-lactate',
                            'Indole-3-acetate', 'Indole-3-carboxaldehyde', 'Indole-3-acrylate',
                            'Indole-3-propionate', 'Indole-3-butyrate', 'Kynurenic acid']]
    data_trp = data_trp.T
    data_trp = data_trp.reset_index()
    data_trp.columns = ['Метаболит', 'Результат']
    data_trp = data4.merge(data_trp, how='left', left_on='Метаболит', right_on='Метаболит')
    df4 = data_trp.copy()
    df4['Вывод'] = df4[df4.columns[1:4]].apply(lambda x: make_result_column(x), axis=1)
    rel_up = df4['Результат'] / df4['Верхняя граница']
    df4['Риск повышения'] = rel_up.loc[rel_up.between(1, 5)]
    df4['Повышено'] = rel_up.loc[rel_up.between(5, np.inf)]
    rel_down = df4['Нижняя граница'] / df4['Результат']
    df4['Риск понижения'] = rel_down.loc[rel_down.between(1, 5)]
    df4['Понижено'] = rel_down.loc[rel_down.between(5, np.inf)]
    df4['Норма'] = df4[['Нижняя граница', 'Результат', 'Вывод']].apply(lambda x: norma(x), axis=1)
    df4 = df4[['Метаболит', 'Нижняя граница', 'Верхняя граница', 'Результат', 'Вывод',
               'Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']]
    color_cols = ['Понижено', 'Риск понижения', 'Норма', 'Риск повышения', 'Повышено']
    st.dataframe(df4.style.applymap(lambda x: color(x), subset=color_cols) \
        .format(na_rep=' ', precision=2, subset=df4.columns) \
        .format(na_rep=' ', precision=0, subset=['Нижняя граница', 'Верхняя граница']))

    st.write('ПРЕДСКАЗАТЕЛЬНАЯ МОДЕЛЬ ОЦЕНКИ РАЗВИТИЯ СЕРДЕЧНО-СОСУДИСТЫХ ЗАБОЛЕВАНИЙ')
    st.write('Классификационная модель построена на основе алгоритма машинного обучения "Случайный Лес"')
    # подгрузка модели
    loaded_model = pickle.load(open('RF_model_1711.pkl', 'rb'))
    #st.write(data_aa.T)
    data1 = pd.read_excel(datazero)
    data1 = data1[
        ['Глицин', 'Аспаргиновая кислота', 'Лизин', 'АДМА', 'АДМА/Аргинин', 'Холин (свободный)', 'Карнитин (С0)',
         'Acetylcarnitine (С2)', 'Propionylcarnitine (С3)',
         'Tiglylcarnitine (C5:1)', 'Isovalerylcarnitine (iC5)', 'Hydroxyisovalerylcarnitine (iC5-OH)',
         'Octenoylcarnitine (C8)', 'Octanoylcarnitine (C8)', 'Adipoylcarnitine (С6DC)',
         'Dodecanoylcarnitine (С12)', 'Tetradecadienoylcarnitine (C14:2)', 'Hydroxytetradecanoylcarnitine (C14:2-OH) ',
         'Hexadecenoylcarnitine (C16:1) ', 'Palmitoylcarnitine (C16)',
         'Hydroxyhexadecenoylcarnitine (C16:1-OH) ', 'Linoleylcarnitine (C18:2)', 'Stearoylcarnitine (C18)',
         '5-Hydroxytryptophan', 'Tryptophan', 'Kynurenine/Tryptophan',
         'Quinolinic acid', 'HIAA', 'Antranillic acid', 'Xanturenic acid', 'Indole-3-lactate', 'Indole-3-acrylate',
         'Kynurenic acid']]
    scaler_sv1 = pickle.load(open('first_scaler1911.pkl', 'rb'))
    data1 = scaler_sv1.transform(data1)
    data1 = preprocessing.normalize(data1, norm='l2')
    a=loaded_model.predict_proba(data1)
    b=float(a[:,1])*100
    st.write('ВЕРОЯТНОСТЬ CCЗ ', round(b,2), '%')
    if b>=70:
        loaded_model2 = pickle.load(open('RF_second_model_1911.pkl', 'rb'))
        scaler_sv2 = pickle.load(open('scaler1911.pkl', 'rb'))
        data2 = pd.read_excel(datazero)
        data2 = data2[['Пролин',
 'Лейцин',
 'Изолейцин',
 'Орнитин',
 'Фенилаланин',
 'Метионин',
 'СДМА',
 'Карнитин (С0)',
 'Glutarylcarnitine (С5DC)',
 'Octenoylcarnitine (C8)',
 'Adipoylcarnitine (С6DC)',
 'Dodecenoylcarnitine (C12:1)',
 'Dodecanoylcarnitine (С12)',
 'Kynurenine',
 'Tryptophan',
 'Kynurenine/Tryptophan',
 'Serotonin',
 'HIAA',
 'Indole-3-carboxaldehyde',
 'Indole-3-acrylate',
 'Indole-3-propionate',
 'Kynurenic acid']]
        data2 = scaler_sv2.transform(data2)
        data2 = preprocessing.normalize(data2, norm='l2')
        c=loaded_model2.predict_proba(data2)
        d=float(c[:,1])*100
        e=float(c[:,0])*100
        st.write('ВЕРОЯТНОСТЬ РАЗВИТИЯ ГБ ', round(d,2), '%')
        st.write('ВЕРОЯТНОСТЬ РАЗВИТИЯ ИБС ', round(e, 2), '%')
   # result1 = pd.read_excel('result1.xlsx')
    #result1['Результат']=round(b,2)
    #if b<70:
    #    result1['Вывод'] = 'Норма'
    #if 70<b<80:
    #    result1['Вывод'] = 'Риск'
    #if b>80:
    #    result1['Вывод'] = 'Повышенный риск'
    #st.dataframe(result1)
    #result2=result1
    #result2['Показатель'] = 'Вероятность НТА'
    #if d>70:
    #    result2['Результат'] = round(d,0)
    #    result2['Вывод'] = 'Риск HTA'
    #else:
    #    result2['Результат'] = round(d,0)
    #    result2['Вывод'] = 'Риск CHD'

    #st.dataframe(result2)