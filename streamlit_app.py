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
st.title("📈 Previsão de Tendência - Ibovespa")

# Sidebar - CONFIGURAÇÕES DE DATA
st.sidebar.header("📅 Período dos Dados")

# Datas padrão: últimos 2 anos
data_fim_padrao = datetime.now()
data_inicio_padrao = data_fim_padrao - timedelta(days=730)  # 2 anos

data_inicio = st.sidebar.date_input(
    "Data Inicial",
    value=data_inicio_padrao,
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now(),
    help="Data inicial para download dos dados"
)

data_fim = st.sidebar.date_input(
    "Data Final",
    value=data_fim_padrao,
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now(),
    help="Data final para download dos dados"
)

# Validação de datas
if data_inicio >= data_fim:
    st.sidebar.error("❌ Data inicial deve ser anterior à data final!")
    st.stop()

dias_diferenca = (data_fim - data_inicio).days
if dias_diferenca < 90:
    st.sidebar.warning(f"⚠️ Período muito curto ({dias_diferenca} dias). Recomendado: mínimo 90 dias")

st.sidebar.info(f"📊 Período selecionado: **{dias_diferenca} dias**")

# Carregar dados com datas personalizadas
@st.cache_data
def load_data(start_date, end_date):
    # Converter para string no formato esperado pelo yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    df = yf.download("^BVSP", start=start_str, end=end_str, multi_level_index=False)
    df.dropna(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    return df

# Baixar dados
with st.spinner(f'📥 Baixando dados do Ibovespa ({data_inicio.strftime("%d/%m/%Y")} a {data_fim.strftime("%d/%m/%Y")})...'):
    df = load_data(data_inicio, data_fim)

# Verificar se há dados suficientes
if len(df) < 60:
    st.error(f"❌ Dados insuficientes ({len(df)} dias). Mínimo recomendado: 60 dias úteis")
    st.stop()

st.success(f"✅ {len(df)} dias de dados carregados com sucesso!")

# Mostrar informações dos dados
with st.expander("ℹ️ Informações dos Dados Carregados"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Primeiro Dia", df.index[0].strftime('%d/%m/%Y'))
    with col2:
        st.metric("Último Dia", df.index[-1].strftime('%d/%m/%Y'))
    with col3:
        st.metric("Total de Dias", len(df))
    with col4:
        variacao_periodo = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        st.metric("Variação no Período", f"{variacao_periodo:+.2f}%")

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
    
    # MACD
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
    
    # Tendência
    df_features['Higher_High'] = (high > high.shift(1)).astype(int)
    df_features['Lower_Low'] = (low < low.shift(1)).astype(int)
    
    # Williams %R
    df_features['Williams_R'] = -100 * ((high_14 - close) / (high_14 - low_14))
    
    # ROC
    for period in [5, 10, 20]:
        df_features[f'ROC_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
    
    df_features.dropna(inplace=True)
    return df_features

# Treinar modelo
def predict_with_optimized_model(df, window_size=30, forecast_days=5, optimize=True):
    df_features = create_advanced_features(df)
    
    train_data = df_features
    
    feature_cols = [col for col in train_data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    
    X = train_data[feature_cols]
    y = train_data['Close'].squeeze()
    
    # Split temporal 85/15
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    st.info(f"📊 Usando {len(X)} amostras | Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    if optimize:
        param_dist = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [8, 10, 12, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf, 
            param_distributions=param_dist, 
            n_iter=15,
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
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        best_rf.fit(X_train, y_train)
        best_params = {}
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # Ensemble
    ensemble = VotingRegressor(
        estimators=[
            ('rf', best_rf),
            ('gb', gb)
        ],
        weights=[0.6, 0.4]
    )
    
    ensemble.fit(X_train, y_train)
    
    # Avaliar
    y_pred_test = ensemble.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Previsões futuras
    predictions = []
    current_full_df = df.copy()
    
    for i in range(forecast_days):
        current_features_df = create_advanced_features(current_full_df)
        last_row = current_features_df[feature_cols].iloc[-1:].copy()
        next_pred = ensemble.predict(last_row)[0]
        predictions.append(next_pred)
        
        last_date = current_full_df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        new_row = pd.DataFrame({
            'Open': [next_pred * 0.999],
            'High': [next_pred * 1.003],
            'Low': [next_pred * 0.997],
            'Close': [next_pred],
            'Volume': [current_full_df['Volume'].iloc[-5:].mean()]
        }, index=[next_date])
        
        current_full_df = pd.concat([current_full_df, new_row])
    
    return predictions, r2, mae, rmse, mape, ensemble, feature_cols, best_params

# Sidebar - CONFIGURAÇÕES DO MODELO
st.sidebar.header("⚙️ Configurações do Modelo")
window_size = st.sidebar.slider("Janela de visualização (dias)", 20, min(90, len(df)), min(60, len(df)))
forecast_days = st.sidebar.slider("Dias a prever", 1, 15, 5)
optimize = st.sidebar.checkbox("Otimizar hiperparâmetros", value=False, help="Melhora R² mas leva 3-5 minutos")

# Botão para treinar
if st.sidebar.button("🚀 Treinar Modelo", type="primary"):
    st.session_state.treinar = True

# Treinar apenas se o botão foi clicado
if 'treinar' not in st.session_state:
    st.info("👈 Configure os parâmetros no menu lateral e clique em **'Treinar Modelo'** para começar")
    st.stop()

# Treinar
if optimize:
    with st.spinner('🔧 Otimizando hiperparâmetros (pode levar alguns minutos)...'):
        predictions, r2, mae, rmse, mape, model, feature_cols, best_params = predict_with_optimized_model(
            df, window_size, forecast_days, optimize
        )
else:
    with st.spinner('⚡ Treinando modelo...'):
        predictions, r2, mae, rmse, mape, model, feature_cols, best_params = predict_with_optimized_model(
            df, window_size, forecast_days, optimize
        )

# Métricas
st.subheader("📊 Métricas de Desempenho do Modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("R² Score", f"{r2:.4f}")

with col2:
    st.metric("MAE", f"R$ {mae:,.2f}")

with col3:
    st.metric("RMSE", f"R$ {rmse:,.2f}")

with col4:
    st.metric("MAPE", f"{mape:.2f}%")

# Interpretar
if r2 >= 0.90:
    st.success(f"🎯 Excelente! R² = {r2:.4f}")
elif r2 >= 0.75:
    st.info(f"✅ Bom ajuste: R² = {r2:.4f}")
elif r2 >= 0.50:
    st.warning(f"⚠️ Ajuste moderado: R² = {r2:.4f}")
elif r2 >= 0:
    st.warning(f"⚠️ Ajuste fraco: R² = {r2:.4f} - Tente otimizar ou usar mais dados")
else:
    st.error(f"❌ R² negativo ({r2:.4f}) - Modelo pior que média simples. Tente período maior de dados")

if optimize and best_params:
    with st.expander("🔍 Melhores Hiperparâmetros"):
        st.json(best_params)

# Gráfico
last_days = df.tail(window_size)
ultimo_valor = df['Close'].iloc[-1]
ultima_data = df.index[-1]

future_dates = pd.bdate_range(start=ultima_data + timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({'Data': future_dates, 'Previsão': predictions})

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
    name=f'Previsão (R²={r2:.3f}, MAPE={mape:.2f}%)',
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
    title=f"Previsão Ibovespa - Ensemble Otimizado (RF + GB)",
    xaxis_title="Data",
    yaxis_title="Valor de Fechamento (R$)",
    hovermode='x unified',
    template='plotly_white',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Tabela
st.subheader("📈 Previsões Detalhadas")

col1, col2 = st.columns(2)

with col1:
    st.metric("Último Fechamento Real", f"R$ {ultimo_valor:,.2f}")

with col2:
    variacao = ((predictions[-1] - ultimo_valor) / ultimo_valor) * 100
    st.metric(f"Previsão {forecast_df['Data'].iloc[-1].strftime('%d/%m/%Y')}", 
              f"R$ {predictions[-1]:,.2f}", f"{variacao:+.2f}%")

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

st.info(f"💡 **Modelo**: Random Forest + Gradient Boosting | {len(feature_cols)} features | Período: {data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}")
