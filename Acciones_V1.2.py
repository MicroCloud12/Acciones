import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df.dropna(inplace=True)
    return df

def create_labels(df, horizon=5):
    df = df.copy()
    future_max = df['Close'].shift(-horizon).rolling(window=horizon).max()
    if isinstance(future_max, pd.DataFrame):
        future_max = future_max.iloc[:, 0]
    if isinstance(df['Close'], pd.DataFrame):
        close_series = df['Close'].iloc[:, 0]
    else:
        close_series = df['Close']
    left, right = future_max.align(close_series, join='inner')
    df = df.loc[left.index]
    df['Target'] = (left > right).astype(int)
    return df

def train_model(df):
    features = ['MACD', 'Signal', 'RSI', 'ATR', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Momentum', 'Stochastic_K']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Modelo entrenado con precisión del: {accuracy*100:.2f}%")
    return model

def generate_signals(df, model):
    features = ['MACD', 'Signal', 'RSI', 'ATR', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Momentum', 'Stochastic_K']
    df['Prediction'] = model.predict(df[features])
    df['Signal'] = 0
    last_entry_index = -10
    holding_period = 3
    signal_col_idx = df.columns.get_loc('Signal')
    for i in range(len(df)):
        pred = df['Prediction'].iloc[i]
        if pred == 1 and (i - last_entry_index) >= holding_period:
            df.iloc[i, signal_col_idx] = 1
            last_entry_index = i
        elif pred == 0 and (i - last_entry_index) >= holding_period:
            df.iloc[i, signal_col_idx] = -1
    return df

def backtest_strategy(df, symbol):
    cash = 10000
    position = 0
    entry_price = 0
    trades = []

    for i, row in df.iterrows():
        signal = row['Signal']
        if isinstance(signal, (pd.Series, np.ndarray)):
            signal = signal.item()
        close_price = row['Close']
        if isinstance(close_price, (pd.Series, np.ndarray)):
            close_price = float(close_price.item())
        fecha = i.strftime('%Y-%m-%d') if isinstance(i, pd.Timestamp) else str(i)
        if signal == 1 and position == 0:
            position = cash / close_price
            entry_price = close_price
            cash = 0
            compra_fecha = fecha
            compra_monto = close_price
        elif signal == -1 and position > 0:
            cash = position * close_price
            position = 0
            venta_fecha = fecha
            venta_monto = close_price
            ganancia = (venta_monto - compra_monto) * (cash / venta_monto)
            trades.append({
                'Compra_Fecha': compra_fecha,
                'Compra_Monto': compra_monto,
                'Venta_Fecha': venta_fecha,
                'Venta_Monto': venta_monto,
                'Ganancia': ganancia
            })

    if position > 0:
        venta_fecha = df.index[-1].strftime('%Y-%m-%d')
        venta_monto = df['Close'].iloc[-1]
        ganancia = (venta_monto - compra_monto) * position
        trades.append({
            'Compra_Fecha': compra_fecha,
            'Compra_Monto': compra_monto,
            'Venta_Fecha': venta_fecha,
            'Venta_Monto': venta_monto,
            'Ganancia': ganancia
        })

    trades_df = pd.DataFrame(trades)
    archivo = f"C:/Users/Mauricio/Documents/Github/Acciones/Operaciones/operaciones_{symbol}.csv"

    # Leer operaciones ya guardadas (si existen)
    if os.path.exists(archivo):
        prev_df = pd.read_csv(archivo, dtype={'Compra_Fecha': str, 'Venta_Fecha': str})
    else:
        prev_df = pd.DataFrame()

    # Filtrar sólo nuevas operaciones (comparando por fecha de compra y venta)
    if not prev_df.empty:
        existentes = set(zip(prev_df['Compra_Fecha'], prev_df['Venta_Fecha']))
    else:
        existentes = set()

    nuevas = trades_df[~trades_df.apply(lambda row: (row['Compra_Fecha'], row['Venta_Fecha']) in existentes, axis=1)]

    # Guardar (append) sólo nuevas operaciones al CSV
    if not nuevas.empty:
        nuevas.to_csv(archivo, mode='a', header=not os.path.exists(archivo), index=False)
    print(f"Nuevas operaciones agregadas a {archivo}: {len(nuevas)}")

    ganancia_total = pd.concat([prev_df, nuevas])['Ganancia'].sum() if not nuevas.empty or not prev_df.empty else 0
    print(f"Ganancia total acumulada para {symbol}: {ganancia_total:.2f} USD")
    return ganancia_total

# Parámetros generales
symbols = ['AAPL', 'MSFT', 'GOOGL','NVDA', 'AMZN', 'META', 'AXP','JPM','V','MA','PYPL','DIS','NFLX','TSLA','INTC','AMD','CSCO','ORCL','IBM','CRM']
start = '2023-01-01'
end = '2025-10-03'

resultados = {}
for symbol in symbols:
    print(f"\nProcesando {symbol} ...")
    df = get_data(symbol, start, end)
    df = add_indicators(df)
    df = create_labels(df, horizon=5)
    model = train_model(df)
    df = generate_signals(df, model)
    ganancia = backtest_strategy(df, symbol)
    resultados[symbol] = ganancia

print("\nResumen de ganancias estimadas por acción:")
for sym, gan in resultados.items():
    print(f"{sym}: {gan:.2f} USD")
