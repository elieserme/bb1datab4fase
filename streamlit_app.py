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

st.dataframe(df)