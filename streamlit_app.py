# --------------------------------------------------------------
# Import library modules for dashboard functionality
# --------------------------------------------------------------

import yfinance as yf
import pandas as pd
import streamlit as st

# --------------------------------------------------------------
# Download and display stock market data
# --------------------------------------------------------------

df = yf.download("^BVSP", period="2y", interval="1d")
df.dropna(inplace=True)

st.title("📈 Análise de índices da Bolsa")

aba1, aba2 = st.tabs(["📈 Gráfico", "📊 Dados brutos"])

with aba1:
    st.write("Gráfico de preços de fechamento do índice Bovespa nos últimos 2 anos")
    st.line_chart(df['Close'])
    
with aba2:
    st.write("Dados brutos")
    rows, columns = df.shape
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Número de linhas", rows)
    with col2:
        st.metric("Número de colunas", columns)    
    st.dataframe(df)