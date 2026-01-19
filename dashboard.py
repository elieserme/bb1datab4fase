import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

st.set_page_config(page_title="Previsão Ibovespa - Random Forest", layout="wide")
st.title("📈 Previsão de Tendência - Ibovespa com Random Forest")

# Carregar dados
@st.cache_data
def load_data():
    # Usar multi_level_index=False para evitar MultiIndex
    df = yf.download("^BVSP", period="2y", interval="1d", multi_level_index=False)
    df.dropna(inplace=True)
    
    # Garantir que as colunas não sejam MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    return df

df = load_data()

# Função para criar features técnicas
def create_features(df, window_size=30):
    df_features = df.copy()
    
    # Garantir que estamos trabalhando com Series, não DataFrames
    close = df_features['Close'].squeeze()
    volume = df_features['Volume'].squeeze()
    high = df_features['High'].squeeze()
    low = df_features['Low'].squeeze()
    
    # Médias Móveis
    df_features['MA_5'] = close.rolling(window=5).mean()
    df_features['MA_10'] = close.rolling(window=10).mean()
    df_features['MA_20'] = close.rolling(window=20).mean()
    
    # Volatilidade
    df_features['Volatility'] = close.rolling(window=10).std()
    
    # Retornos
    df_features['Return_1'] = close.pct_change(1)
    df_features['Return_5'] = close.pct_change(5)
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df_features['Momentum'] = close - close.shift(10)
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df_features['MACD'] = exp1 - exp2
    
    # Lags (valores anteriores)
    for i in range(1, 6):
        df_features[f'Lag_{i}'] = close.shift(i)
    
    # Volume features
    volume_ma = volume.rolling(window=5).mean()
    df_features['Volume_MA'] = volume_ma
    df_features['Volume_Ratio'] = volume / volume_ma
    
    # High-Low range
    df_features['HL_Range'] = high - low
    df_features['HL_Pct'] = (high - low) / close
    
    # Remover NaN
    df_features.dropna(inplace=True)
    
    return df_features

# Função para treinar e prever com Random Forest
def predict_with_rf(df, window_size=30, forecast_days=5, n_estimators=100):
    # Criar features
    df_features = create_features(df)
    
    # Preparar dados de treinamento (últimos 100 dias)
    train_data = df_features.tail(100)
    
    # Features para treinamento
    feature_cols = [col for col in train_data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    
    X = train_data[feature_cols]
    y = train_data['Close'].squeeze()  # Garantir que é Series
    
    # Dividir em treino e teste para calcular R²
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Treinar modelo
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Fazer previsões futuras
    predictions = []
    current_data = df_features.copy()
    
    for i in range(forecast_days):
        # Pegar última linha de features
        last_features = current_data[feature_cols].iloc[-1:].values
        
        # Prever próximo valor
        next_pred = model.predict(last_features)[0]
        predictions.append(next_pred)
        
        # Criar nova linha para próxima previsão
        new_row = current_data.iloc[-1].copy()
        new_row['Close'] = next_pred
        new_row['High'] = next_pred * 1.01
        new_row['Low'] = next_pred * 0.99
        new_row['Volume'] = current_data['Volume'].iloc[-1]
        
        # Adicionar nova linha
        new_row_df = pd.DataFrame([new_row])
        current_data = pd.concat([current_data, new_row_df], ignore_index=False)
        
        # Recalcular features
        current_data = create_features(current_data)
    
    return predictions, r2, mae, rmse, model, feature_cols

# Sidebar
st.sidebar.header("⚙️ Configurações")
window_size = st.sidebar.slider("Janela de análise (dias)", 20, 60, 30)
forecast_days = st.sidebar.slider("Dias a prever", 3, 10, 5)
n_estimators = st.sidebar.slider("Número de árvores (Random Forest)", 50, 200, 100)

# Treinar e prever
with st.spinner('Treinando modelo Random Forest...'):
    predictions, r2, mae, rmse, model, feature_cols = predict_with_rf(
        df, window_size, forecast_days, n_estimators
    )

# Métricas do modelo
st.subheader("📊 Métricas de Desempenho do Modelo")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score",
        value=f"{r2:.4f}",
        help="Quanto mais próximo de 1, melhor o modelo explica a variabilidade dos dados"
    )

with col2:
    st.metric(
        label="MAE (Erro Médio Absoluto)",
        value=f"R$ {mae:,.2f}",
        help="Média dos erros absolutos das previsões"
    )

with col3:
    st.metric(
        label="RMSE (Raiz do Erro Quadrático)",
        value=f"R$ {rmse:,.2f}",
        help="Penaliza mais erros maiores"
    )

# Interpretar R²
if r2 >= 0.8:
    st.success(f"✅ Excelente ajuste do modelo (R² = {r2:.4f})")
elif r2 >= 0.6:
    st.info(f"ℹ️ Bom ajuste do modelo (R² = {r2:.4f})")
elif r2 >= 0.4:
    st.warning(f"⚠️ Ajuste moderado do modelo (R² = {r2:.4f})")
else:
    st.error(f"❌ Ajuste fraco do modelo (R² = {r2:.4f})")

# Preparar dados para visualização
last_days = df.tail(window_size)
ultimo_valor = df['Close'].iloc[-1]
ultima_data = df.index[-1]

# Criar datas futuras
future_dates = pd.bdate_range(start=ultima_data + timedelta(days=1), periods=forecast_days)

# Criar dataframe de previsões
forecast_df = pd.DataFrame({
    'Data': future_dates,
    'Previsão': predictions
})

# Criar gráfico com Plotly
fig = go.Figure()

# Dados históricos
fig.add_trace(go.Scatter(
    x=last_days.index,
    y=last_days['Close'].squeeze(),
    mode='lines+markers',
    name=f'Histórico (últimos {window_size} dias)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5)
))

# Previsões
fig.add_trace(go.Scatter(
    x=forecast_df['Data'],
    y=forecast_df['Previsão'],
    mode='lines+markers',
    name=f'Previsão Random Forest (R²={r2:.3f})',
    line=dict(color='#ff7f0e', width=2, dash='dash'),
    marker=dict(size=8, symbol='diamond')
))

# Linha de conexão
fig.add_trace(go.Scatter(
    x=[ultima_data, forecast_df['Data'].iloc[0]],
    y=[ultimo_valor, predictions[0]],
    mode='lines',
    line=dict(color='gray', width=1, dash='dot'),
    showlegend=False
))

# Layout
fig.update_layout(
    title=f"Previsão Ibovespa - Random Forest ({n_estimators} árvores)",
    xaxis_title="Data",
    yaxis_title="Valor de Fechamento (R$)",
    hovermode='x unified',
    template='plotly_white',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Previsões detalhadas
st.subheader("📈 Previsões Detalhadas")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Último Fechamento Real",
        value=f"R$ {ultimo_valor:,.2f}",
        delta=None
    )

with col2:
    variacao = ((predictions[-1] - ultimo_valor) / ultimo_valor) * 100
    st.metric(
        label=f"Previsão para {forecast_df['Data'].iloc[-1].strftime('%d/%m/%Y')}",
        value=f"R$ {predictions[-1]:,.2f}",
        delta=f"{variacao:+.2f}%"
    )

# Tabela de previsões
st.dataframe(
    pd.DataFrame({
        'Dia': [f"Dia {i+1}" for i in range(forecast_days)],
        'Data': forecast_df['Data'].dt.strftime('%d/%m/%Y'),
        'Valor Previsto': [f"R$ {val:,.2f}" for val in predictions],
        'Tendência': ['🔺 ALTA' if predictions[i] > (ultimo_valor if i == 0 else predictions[i-1]) 
                      else '🔻 BAIXA' for i in range(forecast_days)],
        'Variação %': [f"{((predictions[i] - (ultimo_valor if i == 0 else predictions[i-1])) / (ultimo_valor if i == 0 else predictions[i-1]) * 100):+.2f}%" 
                       for i in range(forecast_days)]
    }),
    use_container_width=True,
    hide_index=True
)

# Importância das features
st.subheader("🎯 Importância das Features")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importância': model.feature_importances_
}).sort_values('Importância', ascending=False).head(10)

fig_importance = go.Figure(go.Bar(
    x=feature_importance['Importância'],
    y=feature_importance['Feature'],
    orientation='h',
    marker=dict(color='#2ca02c')
))

fig_importance.update_layout(
    title="Top 10 Features Mais Importantes",
    xaxis_title="Importância",
    yaxis_title="Feature",
    height=400,
    template='plotly_white'
)

st.plotly_chart(fig_importance, use_container_width=True)

st.warning("⚠️ Esta é uma previsão baseada em Random Forest. O R² Score indica a qualidade do ajuste do modelo aos dados históricos.")
