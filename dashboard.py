import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Previsão Ibovespa - Otimizada", layout="wide")
st.title("📈 Previsão de Tendência - Ibovespa (Modelo Otimizado)")

# Carregar dados
@st.cache_data
def load_data():
    df = yf.download("^BVSP", period="2y", interval="1d", multi_level_index=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    return df

df = load_data()

# Função para criar features técnicas aprimoradas
def create_advanced_features(df):
    df_features = df.copy()
    
    close = df_features['Close'].squeeze()
    volume = df_features['Volume'].squeeze()
    high = df_features['High'].squeeze()
    low = df_features['Low'].squeeze()
    open_price = df_features['Open'].squeeze()
    
    # Médias Móveis (múltiplos períodos)
    for period in [3, 5, 7, 10, 15, 20, 30]:
        df_features[f'MA_{period}'] = close.rolling(window=period).mean()
        df_features[f'MA_Ratio_{period}'] = close / df_features[f'MA_{period}']
    
    # Médias Móveis Exponenciais
    for period in [5, 10, 20]:
        df_features[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # Volatilidade (múltiplos períodos)
    for period in [5, 10, 20, 30]:
        df_features[f'Volatility_{period}'] = close.rolling(window=period).std()
        df_features[f'Volatility_Ratio_{period}'] = df_features[f'Volatility_{period}'] / close
    
    # Retornos (múltiplos períodos)
    for period in [1, 2, 3, 5, 7, 10, 15, 20]:
        df_features[f'Return_{period}'] = close.pct_change(period)
    
    # RSI (múltiplos períodos)
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df_features[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # Momentum
    for period in [5, 10, 20]:
        df_features[f'Momentum_{period}'] = close - close.shift(period)
    
    # MACD (diferentes configurações)
    exp1_12 = close.ewm(span=12, adjust=False).mean()
    exp2_26 = close.ewm(span=26, adjust=False).mean()
    df_features['MACD'] = exp1_12 - exp2_26
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    df_features['MACD_Diff'] = df_features['MACD'] - df_features['MACD_Signal']
    
    # Bollinger Bands
    for period in [10, 20]:
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        df_features[f'BB_Upper_{period}'] = ma + (2 * std)
        df_features[f'BB_Lower_{period}'] = ma - (2 * std)
        df_features[f'BB_Width_{period}'] = (df_features[f'BB_Upper_{period}'] - df_features[f'BB_Lower_{period}']) / ma
        df_features[f'BB_Position_{period}'] = (close - df_features[f'BB_Lower_{period}']) / (df_features[f'BB_Upper_{period}'] - df_features[f'BB_Lower_{period}'])
    
    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df_features['Stochastic'] = 100 * ((close - low_14) / (high_14 - low_14))
    
    # Average True Range (ATR)
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_features['ATR'] = true_range.rolling(14).mean()
    
    # Lags (valores anteriores)
    for i in range(1, 11):
        df_features[f'Lag_{i}'] = close.shift(i)
    
    # Volume features
    for period in [5, 10, 20]:
        df_features[f'Volume_MA_{period}'] = volume.rolling(window=period).mean()
        df_features[f'Volume_Ratio_{period}'] = volume / df_features[f'Volume_MA_{period}']
    
    # Price patterns
    df_features['HL_Range'] = high - low
    df_features['HL_Pct'] = (high - low) / close
    df_features['OC_Range'] = close - open_price
    df_features['OC_Pct'] = (close - open_price) / open_price
    
    # Tendência de alta/baixa
    df_features['Higher_High'] = (high > high.shift(1)).astype(int)
    df_features['Lower_Low'] = (low < low.shift(1)).astype(int)
    
    # Williams %R
    df_features['Williams_R'] = -100 * ((high_14 - close) / (high_14 - low_14))
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df_features[f'ROC_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
    
    # Remover NaN
    df_features.dropna(inplace=True)
    
    return df_features

# Função para treinar com otimização de hiperparâmetros e ensemble
def predict_with_optimized_model(df, window_size=30, forecast_days=5, optimize=True):
    # Criar features avançadas
    df_features = create_advanced_features(df)
    
    # Preparar dados (usar mais dados para treino)
    train_data = df_features.tail(200)
    
    feature_cols = [col for col in train_data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    
    X = train_data[feature_cols]
    y = train_data['Close'].squeeze()
    
    # Dividir em treino e teste
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if optimize:
        # Otimização de hiperparâmetros com RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf, 
            param_distributions=param_dist, 
            n_iter=20,
            cv=3, 
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_
    else:
        best_rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        best_rf.fit(X_train, y_train)
        best_params = {}
    
    # Criar ensemble com múltiplos modelos
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    )
    
    # Ensemble voting
    ensemble = VotingRegressor(
        estimators=[
            ('rf', best_rf),
            ('gb', gb)
        ],
        weights=[0.6, 0.4]
    )
    
    ensemble.fit(X_train, y_train)
    
    # Avaliar modelos
    y_pred_test = ensemble.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Fazer previsões futuras
    predictions = []
    current_data = df_features.copy()
    
    for i in range(forecast_days):
        last_features = current_data[feature_cols].iloc[-1:].values
        next_pred = ensemble.predict(last_features)[0]
        predictions.append(next_pred)
        
        # Criar nova linha
        new_row = current_data.iloc[-1].copy()
        new_row['Close'] = next_pred
        new_row['High'] = next_pred * 1.005
        new_row['Low'] = next_pred * 0.995
        new_row['Open'] = next_pred * 0.998
        new_row['Volume'] = current_data['Volume'].iloc[-5:].mean()
        
        new_row_df = pd.DataFrame([new_row])
        current_data = pd.concat([current_data, new_row_df], ignore_index=False)
        current_data = create_advanced_features(current_data)
    
    return predictions, r2, mae, rmse, ensemble, feature_cols, best_params

# Sidebar
st.sidebar.header("⚙️ Configurações")
window_size = st.sidebar.slider("Janela de análise (dias)", 20, 60, 30)
forecast_days = st.sidebar.slider("Dias a prever", 3, 10, 5)
optimize = st.sidebar.checkbox("Otimizar hiperparâmetros", value=True, help="Melhora R² mas leva mais tempo")

# Treinar e prever
if optimize:
    with st.spinner('🔧 Otimizando hiperparâmetros e treinando ensemble...'):
        predictions, r2, mae, rmse, model, feature_cols, best_params = predict_with_optimized_model(
            df, window_size, forecast_days, optimize
        )
else:
    with st.spinner('⚡ Treinando modelo rápido...'):
        predictions, r2, mae, rmse, model, feature_cols, best_params = predict_with_optimized_model(
            df, window_size, forecast_days, optimize
        )

# Métricas do modelo
st.subheader("📊 Métricas de Desempenho do Modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score",
        value=f"{r2:.4f}",
        help="Quanto mais próximo de 1, melhor"
    )

with col2:
    st.metric(
        label="MAE",
        value=f"R$ {mae:,.2f}"
    )

with col3:
    st.metric(
        label="RMSE",
        value=f"R$ {rmse:,.2f}"
    )

# Interpretar R²
if r2 >= 0.9:
    st.success(f"🎯 Excelente ajuste do modelo (R² = {r2:.4f})")
elif r2 >= 0.7:
    st.info(f"✅ Bom ajuste do modelo (R² = {r2:.4f})")
elif r2 >= 0.5:
    st.warning(f"⚠️ Ajuste moderado do modelo (R² = {r2:.4f})")
else:
    st.error(f"❌ Ajuste fraco - considere otimizar (R² = {r2:.4f})")

# Mostrar melhores parâmetros se otimizado
if optimize and best_params:
    with st.expander("🔍 Melhores Hiperparâmetros Encontrados"):
        st.json(best_params)

# Preparar visualização
last_days = df.tail(window_size)
ultimo_valor = df['Close'].iloc[-1]
ultima_data = df.index[-1]

future_dates = pd.bdate_range(start=ultima_data + timedelta(days=1), periods=forecast_days)

forecast_df = pd.DataFrame({
    'Data': future_dates,
    'Previsão': predictions
})

# Gráfico
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=last_days.index,
    y=last_days['Close'].squeeze(),
    mode='lines+markers',
    name=f'Histórico ({window_size} dias)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5)
))

fig.add_trace(go.Scatter(
    x=forecast_df['Data'],
    y=forecast_df['Previsão'],
    mode='lines+markers',
    name=f'Previsão Ensemble (R²={r2:.3f})',
    line=dict(color='#2ca02c', width=3, dash='dash'),
    marker=dict(size=10, symbol='diamond')
))

fig.add_trace(go.Scatter(
    x=[ultima_data, forecast_df['Data'].iloc[0]],
    y=[ultimo_valor, predictions[0]],
    mode='lines',
    line=dict(color='gray', width=1, dash='dot'),
    showlegend=False
))

fig.update_layout(
    title=f"Previsão Ibovespa - Modelo Ensemble Otimizado",
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
        value=f"R$ {ultimo_valor:,.2f}"
    )

with col2:
    variacao = ((predictions[-1] - ultimo_valor) / ultimo_valor) * 100
    st.metric(
        label=f"Previsão para {forecast_df['Data'].iloc[-1].strftime('%d/%m/%Y')}",
        value=f"R$ {predictions[-1]:,.2f}",
        delta=f"{variacao:+.2f}%"
    )

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

st.info("💡 **Modelo Ensemble**: Combina Random Forest otimizado + Gradient Boosting para maior precisão")
st.warning("⚠️ Previsões são estimativas estatísticas. Não devem ser usadas como única base para investimentos.")
