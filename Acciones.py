import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    # Bandas de Bollinger
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    # Estocástico %K (14)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)

    df.dropna(inplace=True)
    return df

def create_labels(df, horizon=5):
    df = df.copy()
    future_max = df['Close'].shift(-horizon).rolling(window=horizon).max()

    # Extraer series unidimensionales si es DataFrame
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
    print(f"Modelo mejorado entrenado con precisión del: {accuracy*100:.2f}%")
    return model

def generate_signals(df, model):
    features = ['MACD', 'Signal', 'RSI', 'ATR', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Momentum', 'Stochastic_K']
    df['Prediction'] = model.predict(df[features])
    df['Signal'] = 0
    last_entry_index = -10  # índice fuera de rango para inicio
    holding_period = 3
    signal_col_idx = df.columns.get_loc('Signal')

    for i in range(len(df)):
        pred = df['Prediction'].iloc[i]
        if pred == 1 and (i - last_entry_index) >= holding_period:
            df.iloc[i, signal_col_idx] = 1  # Comprar
            last_entry_index = i
        elif pred == 0 and (i - last_entry_index) >= holding_period:
            df.iloc[i, signal_col_idx] = -1  # Vender

    return df

def backtest_strategy(df):
    cash = 10000
    position = 0
    entry_price = 0

    for i, row in df.iterrows():
        signal = row['Signal']
        if isinstance(signal, (pd.Series, np.ndarray)):
            signal = signal.item()
        close_price = row['Close']
        if isinstance(close_price, (pd.Series, np.ndarray)):
            close_price = float(close_price.item())
        if signal == 1 and position == 0:
            position = cash / close_price
            entry_price = close_price
            cash = 0
            fecha = i.strftime('%Y-%m-%d') if isinstance(i, pd.Timestamp) else str(i)
            print(f"Compra en {fecha} a {entry_price:.2f}")
        elif signal == -1 and position > 0:
            cash = position * close_price
            position = 0
            gain = (close_price - entry_price) * (cash / entry_price)
            fecha = i.strftime('%Y-%m-%d') if isinstance(i, pd.Timestamp) else str(i)
            print(f"Venta en {fecha} a {close_price:.2f}, ganancia: {gain:.2f}")

    if position > 0:
        final_value = position * df['Close'].iloc[-1]
    else:
        final_value = cash
    ganancia = final_value - 10000
    print(f"Ganancia total estimada: {ganancia:.2f} USD")
    return ganancia



# Uso
symbol = 'AAPL'
start = '2023-01-01'
end = '2025-09-15'

df = get_data(symbol, start, end)
df = add_indicators(df)
df = create_labels(df, horizon=5)
model = train_model(df)
df = generate_signals(df, model)
ganancia = backtest_strategy(df)

print("Primeras 5 fechas para entrar:")
print(df[df['Signal'] == 1].index.strftime('%Y-%m-%d').tolist()[:5])

print("Primeras 5 fechas para salir:")
print(df[df['Signal'] == -1].index.strftime('%Y-%m-%d').tolist()[:5])
