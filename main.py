from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
# from statsforecast import StatsForecast
# from statsforecast.models import Naive, MSTL
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

import streamlit as st

st.title('Análise para predição dos valores de fechamento do índice Bovespa')

tab_0, tab_1 = st.tabs(['Análise Exploratória', 'Modelo'])

df = pd.read_csv('Dados Históricos - Ibovespa.csv')

df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df = df[['Data', 'Último']]
df.columns = ['ds', 'y']
df = df.set_index('ds')

ad_fuller = adfuller(df.y.values)
p_value = ad_fuller[1]

with tab_0:
    """
    O IBOVESPA (Índice da Bolsa de Valores de São Paulo) é o principal índice de ações do mercado de capitais brasileiro e 
    serve como indicador do desempenho médio das cotações das ações mais negociadas e mais representativas do mercado brasileiro. 
    Ele é utilizado tanto para entender o comportamento do mercado acionário brasileiro como um todo, quanto como referência 
    para investimentos. Um índice forte pode indicar um mercado em alta, com crescimento econômico e confiança dos investidores, 
    enquanto um índice fraco pode sinalizar o contrário.

    Antes de predizer o fechamos da base, devemos entender o contexto inserido. Qual o comportamento da serie em questao, e identificar
    é uma serie estacionaria, ou nao-estacionaria. No grafico abaixo, podemos visualizar - em um primeiro momento - o comportamento dos dados
    ao longo dos anos.
    """

    plt.figure()
    sns.lineplot(data=df, x='ds', y='y')
    plt.xlabel('Ano')
    plt.ylabel('Valor (R$) (em milhares)')
    plt.title('Índice Bovespa')
    st.pyplot(plt)

    """
    Para esta analise, extraimos 20 anos de dados. Esse conjunto foi suficiente para entender o comportamento da bolsa de valores IBOVESPA e 
    entender as principais tendencias. Conforme grafico acima, podemos observar que em linhas gerais temos uma tendencia de aumento na bolsa dese 2004.
    Entretanto a bolsa de valores foi marcada por uma series de fatores, e foi diretamente impactada por eles. 

    
    """
    st.image("https://www.b3.com.br/data/files/42/20/55/D4/E0AB8810C7AB8988AC094EA8/Linha%20do%20Tempo%20Ibovespa%20B3.png")
    """
    Decompondo a série temporal, para uma sazonalidade de 1 ano:
    """

    plt.figure()
    resultados = seasonal_decompose(df, period=247)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    resultados.observed.plot(ax=axes[0])
    resultados.trend.plot(ax=axes[1])
    resultados.seasonal.plot(ax=axes[2])
    resultados.resid.plot(ax=axes[3])
    plt.tight_layout()
    st.pyplot(plt)

    f"""
    Aplicando-se os teste de Dickey-Fuller, temos um valor de P-value = {p_value}.
    Conclusão: série não estácionária.
    """


def calcula_wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


df = df.reset_index('ds')
df['unique_id'] = 0
df_treino = df[df.ds < '2022-08-07']
df_valid = df[df.ds >= '2022-08-07']
# h = len(df_valid['ds'])


# model = StatsForecast(models=[MSTL(season_length=[247, 22, 5], trend_forecaster=Naive())], freq='B', n_jobs=-1)
# model.fit(df_treino)
# joblib.dump(model, 'model.joblib')


with tab_1:
    h = st.slider("Selecione um valor entre 50 e 260, para a predição:", 50, 260, value=155)
    model = joblib.load('model.joblib')
    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(df_valid, on=['ds', 'unique_id'], how='left')
    forecast_df = forecast_df.dropna()

    wmape2 = calcula_wmape(forecast_df['y'].values, forecast_df['MSTL'].values)

    plt.figure(figsize=(20, 6))
    sns.lineplot(data=df_treino[df_treino.ds >= '2019-05-01'], x='ds', y='y', color='b', label='Real')
    sns.lineplot(data=forecast_df, x='ds', y='y', color='b')
    sns.lineplot(data=forecast_df, x='ds', y='MSTL', color='g', label='Predito')
    sns.lineplot(data=forecast_df, x='ds', y='MSTL-lo-90', color='r', label='Intervalo de confianca')
    sns.lineplot(data=forecast_df, x='ds', y='MSTL-hi-90', color='r')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    f"""
    WMAPE para o período fornecido: {wmape2:.2%}
    """
