from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

app = Flask(__name__)

API_KEY = 'FVOEWU64HKN1C9U2'
STOCK_BASE_URL = 'https://www.alphavantage.co/query'
HOLIDAY_API_KEY = '49339829-1b08-49a6-b341-72f937bb885f'
HOLIDAY_API_URL = 'https://holidayapi.com/v1/holidays'

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

def train_random_forest(df):
    df = add_technical_indicators(df)

    # Features and target variable
    X = df[['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
    y = df['Close'].shift(-1)  # Predict the next day's closing price

    # Remove the last row with NaN target
    X = X[:-1]
    y = y[:-1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler

def fetch_stock_data(symbol):
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

        return df

    return None

def plot_prices(dates, predicted_prices, actual_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='blue', marker='o')
    plt.plot(dates, actual_prices, label='Actual Prices', color='green', marker='x')
    plt.title('Stock Prices: Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plot_filename = 'static/plot.png'
    if os.path.exists(plot_filename):
        os.remove(plot_filename)  # Remove the old plot if it exists
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

def is_holiday(date, country='US'):
    params = {
        'key': HOLIDAY_API_KEY,
        'country': country,
        'year': date.year,
        'month': date.month,
        'day': date.day,
    }
    response = requests.get(HOLIDAY_API_URL, params=params)
    holidays = response.json().get('holidays', [])
    return len(holidays) > 0

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_prices = []
    actual_prices = []
    error_message = None
    future_dates = []
    future_prediction = None
    accuracy_score = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        future_date_str = request.form.get('future_date', '')

        # Fetch the stock data from Alpha Vantage
        df = fetch_stock_data(symbol)

        if df is not None:
            # Train the model
            model, scaler = train_random_forest(df)

            if future_date_str:
                future_date = pd.to_datetime(future_date_str)

                # Check if the entered date is a holiday
                if is_holiday(future_date):
                    error_message = f"There is a holiday on {future_date.strftime('%Y-%m-%d')}. No prediction available."
                    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                                           accuracy_score=accuracy_score)

                last_date = df.index[-1]

                if future_date > last_date:
                    # Prepare data for future date prediction
                    last_row = df.iloc[-1][['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
                    last_row_df = pd.DataFrame([last_row])
                    last_row_scaled = scaler.transform(last_row_df)

                    # Predict the price for the future date
                    future_prediction = model.predict(last_row_scaled)[0]

                    # Print the predicted value for the user-entered future date
                    print(f"The prediction for {future_date.strftime('%Y-%m-%d')}, is {future_prediction}")

            # Generate dates from January 1st to today
            start_date = datetime(2024, 1, 1)
            end_date = datetime.now()
            date_range = pd.date_range(start=start_date, end=end_date)

            # Prepare data for predictions
            for date in date_range:
                if date in df.index:
                    last_row = df.loc[date][['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
                    last_row_df = pd.DataFrame([last_row])
                    last_row_scaled = scaler.transform(last_row_df)

                    # Predict the next day's price
                    predicted_price = model.predict(last_row_scaled)[0]
                    predicted_prices.append(predicted_price)
                    actual_prices.append(df.loc[date]['Close'])
                    future_dates.append(date)

            # Calculate accuracy score
            accuracy_score = r2_score(actual_prices, predicted_prices)*100

            # Plot the prices
            plot_filename = plot_prices(future_dates, predicted_prices, actual_prices)

            return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                   future_dates=future_dates, plot_url=plot_filename, future_prediction=future_prediction,
                                   accuracy_score=accuracy_score)

        else:
            error_message = "Failed to fetch stock data. Please check the stock symbol."

    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                           accuracy_score=accuracy_score)

if __name__ == '__main__':
    app.run(debug=True)