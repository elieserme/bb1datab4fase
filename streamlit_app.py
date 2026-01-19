# --------------------------------------------------------------
# Import library modules for dashboard functionality
# --------------------------------------------------------------

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# --------------------------------------------------------------
# Download and display stock market data
# --------------------------------------------------------------



st.title("📈 Análise de índices da Bolsa")
st.write("Participantes: Andrea Grassmann, Victor Almeida e Eliéser Reis")
ticker_symbol = st.text_input("Digite o índice (ex: ^BVSP)", "^BVSP")
entry1, entry2 = st.columns(2)
with entry1:
    start_date = st.date_input("Data inicial (sugerido 2 anos atrás)", datetime.now() - timedelta(days=365*2), format="DD/MM/YYYY")
with entry2:
    end_date = st.date_input("Data final", pd.to_datetime("today"), format="DD/MM/YYYY")
df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d")
# df['Date'] = pd.to_datetime(df.index)
# df.dropna(inplace=True)
df.sort_values(by='Date', inplace=True, ascending=False)

aba1, aba2 = st.tabs(["📈 Gráfico", "📊 Dados brutos"])

with aba1:
    st.write(f"Gráfico de preços de fechamento do índice {ticker_symbol} de {start_date} até {end_date}")
    st.line_chart(df['Close'])
    # cut_off_date = end_date - pd.Timedelta(days=30)
    # filtered_df = df[df['Date'] > cut_off_date]
    # st.write(f"Gráfico de preços de fechamento do índice {ticker_symbol} dos últimos 30 dias")
    # st.line_chart(filtered_df['Close'])
    
with aba2:
    st.write("Dados brutos")
    rows, columns = df.shape
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Número de linhas", rows)
    with col2:
        st.metric("Número de colunas", columns)    
    st.dataframe(df)