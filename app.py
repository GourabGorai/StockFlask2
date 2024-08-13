from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import requests
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

API_KEY = 'FVOEWU64HKN1C9U2'
EXCHANGE_RATE_API_KEY = '699cc6dc962264a2fc44c056'
STOCK_BASE_URL = 'https://www.alphavantage.co/query'
EXCHANGE_RATE_BASE_URL = 'https://v6.exchangerate-api.com/v6'


def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df


def add_technical_indicators(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['High'] - df['Low']
    df['RSI'] = calculate_rsi(df)
    df = calculate_macd(df)
    df.fillna(0, inplace=True)
    return df


def train_decision_tree(df, currency='USD'):
    df = add_technical_indicators(df)

    X = df[['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
    y = df['Close']

    if currency == 'INR':
        X = df[['Close_INR', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
        y = df['Close_INR']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    return model, scaler


def fetch_future_price_from_api(symbol, future_date, api_key):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }

    response = requests.get(STOCK_BASE_URL, params=params)
    data = response.json()

    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        future_date_str = future_date.strftime('%Y-%m-%d')
        if future_date_str in time_series:
            actual_price = time_series[future_date_str]['4. close']
            return float(actual_price)

    return None


def plot_prices(future_date, predicted_price, actual_price=None):
    dates = [future_date]
    prices = [predicted_price]
    labels = ['Predicted']

    if actual_price is not None:
        prices.append(actual_price)
        labels.append('Actual')

    plt.figure(figsize=(10, 6))
    plt.bar(labels, prices, color=['blue', 'green'])
    plt.title(f'Stock Price on {future_date}')
    plt.ylabel('Price')
    plt.xlabel('Type')

    plot_filename = 'static/plot.png'
    if os.path.exists(plot_filename):
        os.remove(plot_filename)  # Remove the old plot if it exists
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        currency = request.form['currency'].upper()
        future_date_str = request.form['future_date']
        future_date = datetime.strptime(future_date_str, '%Y-%m-%d')

        # Step 1: Fetch the exchange rate (USD to INR)
        exchange_rate_response = requests.get(f"{EXCHANGE_RATE_BASE_URL}/{EXCHANGE_RATE_API_KEY}/latest/USD")
        exchange_rate_data = exchange_rate_response.json()
        usd_to_inr = exchange_rate_data['conversion_rates']['INR']

        # Step 2: Fetch the stock data from Alpha Vantage
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': API_KEY
        }

        response = requests.get(STOCK_BASE_URL, params=params)
        data = response.json()

        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)

            df['Open_INR'] = df['Open'] * usd_to_inr
            df['High_INR'] = df['High'] * usd_to_inr
            df['Low_INR'] = df['Low'] * usd_to_inr
            df['Close_INR'] = df['Close'] * usd_to_inr

            # Step 3: Train the model
            model, scaler = train_decision_tree(df, currency=currency)

            # Step 4: Predict the stock price for the future date
            last_row = df.iloc[-1][
                ['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD',
                 'MACD_signal']] if currency == 'USD' else \
                df.iloc[-1][['Close_INR', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]

            last_row_df = pd.DataFrame([last_row])
            last_row_scaled = scaler.transform(last_row_df)
            predicted_price = model.predict(last_row_scaled)[0]

            # Step 5: Fetch the actual price from the API for the future date
            actual_price = fetch_future_price_from_api(symbol, future_date, API_KEY)

            # Step 6: Plot the prices
            plot_filename = plot_prices(future_date_str, predicted_price, actual_price)

            return render_template('index.html', predicted_price=predicted_price, future_date=future_date_str,
                                   currency=currency, actual_price=actual_price, plot_url=plot_filename)

        else:
            error_message = data.get('Error Message') or data.get('Note') or data.get(
                'Information') or "An unknown error occurred."
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
