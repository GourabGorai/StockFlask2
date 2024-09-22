from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import plotly.express as px
import plotly.io as pio
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
    df.loc[:, 'EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df.loc[:, 'EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df.loc[:, 'MACD'] = df['EMA_fast'] - df['EMA_slow']
    df.loc[:, 'MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df

def add_technical_indicators(df):
    df.loc[:, 'MA_5'] = df['Close'].rolling(window=5).mean()
    df.loc[:, 'MA_10'] = df['Close'].rolling(window=10).mean()
    df.loc[:, 'MA_50'] = df['Close'].rolling(window=50).mean()
    df.loc[:, 'Volatility'] = df['High'] - df['Low']
    df.loc[:, 'RSI'] = calculate_rsi(df)
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
    fig = px.line(
        x=dates,
        y=[predicted_prices, actual_prices],
        labels={'x': 'Date', 'y': 'Price'}
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        title='Stock Prices: Predicted vs Actual',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.data[0].name = 'Predicted Prices'
    fig.data[1].name = 'Actual Prices'

    # Update hover template to show only "date" and "value"
    hover_template = "<b>Date:</b> %{x}<br><b>Price:</b> %{y}"
    for trace in fig.data:
        trace.hovertemplate = hover_template

    plot_filename = 'static/plot.html'
    pio.write_html(fig, file=plot_filename, auto_open=False)

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
        year_prv = datetime.now().year - 1
        # Fetch the stock data from Alpha Vantage
        df = fetch_stock_data(symbol)

        if df is not None:
            # Apply technical indicators
            df = add_technical_indicators(df)
            df2 = df[df.index <= f'{year_prv}-12-29']  # Use data until 2023-12-29

            # Train the model
            model, scaler = train_random_forest(df2)

            if future_date_str:
                future_date = pd.to_datetime(future_date_str)
                last_date = df.index[-1]

                if future_date <= last_date or is_holiday(future_date):
                    error_message = "No prediction available."
                    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                                           accuracy_score=accuracy_score)

                # Simulate future data for the given future date
                future_df = df.copy()
                current_date = last_date

                while current_date < future_date:
                    next_row = future_df.iloc[-1][['Close', 'MA_5', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_signal']]
                    next_row_df = pd.DataFrame([next_row])
                    next_row_scaled = scaler.transform(next_row_df)

                    # Predict the next day's closing price
                    predicted_price = model.predict(next_row_scaled)[0]

                    # Update the future_df with the predicted price and recalculate indicators
                    next_date = current_date + timedelta(days=1)
                    new_row = pd.Series({
                        'Open': predicted_price,
                        'High': predicted_price,
                        'Low': predicted_price,
                        'Close': predicted_price,
                        'Volume': 0,
                        'MA_5': 0,
                        'MA_10': 0,
                        'MA_50': 0,
                        'Volatility': 0,
                        'RSI': 0,
                        'MACD': 0,
                        'MACD_signal': 0,
                    }, name=next_date)

                    future_df = pd.concat([future_df, new_row.to_frame().T])
                    future_df = add_technical_indicators(future_df)
                    current_date = next_date

                future_prediction = round(predicted_price, 2)
                print(f"The prediction for {future_date_str} is {future_prediction}")

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
            accuracy_score = round(r2_score(actual_prices, predicted_prices) * 100, 2)

            # Plot the prices
            plot_filename = plot_prices(future_dates, predicted_prices, actual_prices)

            return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                                   future_dates=future_dates, plot_url=plot_filename, future_prediction=future_prediction,
                                   accuracy_score=accuracy_score)

        else:
            error_message = "Failed to fetch stock data. Please try again."

    return render_template('index.html', predicted_prices=predicted_prices, actual_prices=actual_prices,
                           future_dates=future_dates, error_message=error_message, future_prediction=future_prediction,
                           accuracy_score=accuracy_score)

if __name__ == '__main__':
    app.run(debug=True)
