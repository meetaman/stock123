from flask import Flask, render_template, request, redirect, url_for, Response
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
import time

app = Flask(__name__)

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['MA'] + (data['STD'] * num_std)
    data['Lower_Band'] = data['MA'] - (data['STD'] * num_std)
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate candlestick patterns manually
def calculate_candlestick_patterns(data):
    # Doji: Open and Close are nearly equal
    data['Doji'] = np.where(abs(data['Open'] - data['Close']) <= 0.01 * data['Close'], 1, 0)

    # Bullish Engulfing: Current candle fully engulfs the previous candle
    data['Bullish_Engulfing'] = np.where(
        (data['Close'] > data['Open']) &  # Current candle is bullish
        (data['Open'] < data['Close'].shift(1)) &  # Current open < previous close
        (data['Close'] > data['Open'].shift(1)),  # Current close > previous open
        1, 0
    )

    # Bearish Engulfing: Current candle fully engulfs the previous candle
    data['Bearish_Engulfing'] = np.where(
        (data['Close'] < data['Open']) &  # Current candle is bearish
        (data['Open'] > data['Close'].shift(1)) &  # Current open > previous close
        (data['Close'] < data['Open'].shift(1)),  # Current close < previous open
        1, 0
    )

    # Hammer: Small body, long lower wick, little to no upper wick
    body = abs(data['Close'] - data['Open'])
    lower_wick = data['Open'] - data['Low']
    upper_wick = data['High'] - data['Close']
    data['Hammer'] = np.where(
        (body <= 0.02 * data['Close']) &  # Small body
        (lower_wick >= 2 * body) &  # Long lower wick
        (upper_wick <= 0.1 * body),  # Little to no upper wick
        1, 0
    )

    # Shooting Star: Small body, long upper wick, little to no lower wick
    data['Shooting_Star'] = np.where(
        (body <= 0.02 * data['Close']) &  # Small body
        (upper_wick >= 2 * body) &  # Long upper wick
        (lower_wick <= 0.1 * body),  # Little to no lower wick
        1, 0
    )

    return data

# Function to generate a reason for the recommendation
def generate_reason(data):
    latest_data = data.iloc[-1]
    reasons = []

    # RSI Analysis
    if latest_data['RSI'] < 30:
        reasons.append("RSI indicates oversold conditions.")
    elif latest_data['RSI'] > 70:
        reasons.append("RSI indicates overbought conditions.")

    # MACD Analysis
    if latest_data['MACD'] > latest_data['Signal_Line']:
        reasons.append("MACD indicates bullish momentum.")
    else:
        reasons.append("MACD indicates bearish momentum.")

    # Candlestick Patterns
    if latest_data['Doji'] == 1:
        reasons.append("Doji pattern detected, indicating potential reversal.")
    if latest_data['Bullish_Engulfing'] == 1:
        reasons.append("Bullish Engulfing pattern detected, indicating potential reversal.")
    if latest_data['Bearish_Engulfing'] == 1:
        reasons.append("Bearish Engulfing pattern detected, indicating potential reversal.")
    if latest_data['Hammer'] == 1:
        reasons.append("Hammer pattern detected, indicating potential bullish reversal.")
    if latest_data['Shooting_Star'] == 1:
        reasons.append("Shooting Star pattern detected, indicating potential bearish reversal.")

    # Combine reasons into a single string
    return " ".join(reasons)

def fetch_and_preprocess_data(ticker, period='3mo', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            print(f"No data found for {ticker}. Skipping...")
            return None

        # Feature Engineering
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()

        data = calculate_bollinger_bands(data, window=20, num_std=2)
        data = calculate_rsi(data, window=14)
        data = calculate_macd(data, short_window=12, long_window=26, signal_window=9)
        data = calculate_candlestick_patterns(data)

        data['Volume_MA'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

        data['Price_Change'] = data['Close'].pct_change()

        # Ensure we have some variation in the spike labels
        volume_threshold = np.percentile(data['Volume_Ratio'].dropna(), 95)  # Use 95th percentile as threshold
        data['Spike'] = np.where(data['Volume_Ratio'] > volume_threshold, 1, 0)

        data.dropna(inplace=True)

        # Check if we have both classes (0 and 1) in the Spike column
        if len(data['Spike'].unique()) < 2:
            print(f"Warning: {ticker} doesn't have enough variation in volume spikes")
            return None

        return data
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def train_and_evaluate_model(data):
    features = ['Volume_Ratio', 'Price_Change', 'MA_5', 'MA_20', 'MA_50', 'Upper_Band', 'Lower_Band', 'RSI', 'MACD', 'Signal_Line', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer', 'Shooting_Star']
    X = data[features]
    y = data['Spike']

    # Check if we have enough samples for both classes
    if len(y.unique()) < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if we have both classes in training data
    if len(np.unique(y_train)) < 2:
        return None

    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)

    return model

def predict_spike_probability(model, data):
    if model is None:
        return 0.0

    features = ['Volume_Ratio', 'Price_Change', 'MA_5', 'MA_20', 'MA_50', 'Upper_Band', 'Lower_Band', 'RSI', 'MACD', 'Signal_Line', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer', 'Shooting_Star']
    latest_data = data[features].iloc[-1:].copy()
    latest_data = latest_data[features]
    
    try:
        # Get probability predictions
        probabilities = model.predict_proba(latest_data)
        
        # Check if we have probabilities for both classes
        if probabilities.shape[1] >= 2:
            return probabilities[0][1]  # Return probability for class 1
        else:
            return 0.0  # Return 0 if only one class is predicted
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.0

# SSE route to stream logs
@app.route('/stream')
def stream():
    def generate():
        with open(f"log_{timestamp}.txt", "r") as log_file:
            while True:
                line = log_file.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/', methods=['GET', 'POST'])
def index():
    global timestamp
    if request.method == 'POST':
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}.csv"
        log_filename = f"log_{timestamp}.txt"

        # Set up logging
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

        # Read stock symbols from text file
        with open('saaf.txt', 'r') as file:
            stock_symbols = [line.strip() for line in file.readlines()]

        results = []
        for symbol in stock_symbols:
            logging.info(f"Processing {symbol}...")
            data = fetch_and_preprocess_data(symbol, period='3mo', interval='1d')
            
            if data is not None:
                model = train_and_evaluate_model(data)
                if model is not None:
                    probability = predict_spike_probability(model, data)
                    reason = generate_reason(data)
                    results.append({'Stock': symbol, 'Spike_Probability': probability, 'Reason': reason})
                    logging.info(f"Processed {symbol}: Probability = {probability}, Reason = {reason}")
                else:
                    logging.warning(f"Could not train model for {symbol}")

        if results:
            results_df = pd.DataFrame(results)
            results_df['Rank'] = results_df['Spike_Probability'].rank(ascending=False, method='min').astype(int)
            results_df = results_df.sort_values(by='Rank')
            results_df.to_csv(csv_filename, index=False)
            logging.info(f"Results saved to {csv_filename}")
            return redirect(url_for('results', timestamp=timestamp))
        else:
            logging.warning("No valid results to save.")
            return "No valid results to save."

    return render_template('index.html')

@app.route('/results')
def results():
    timestamp = request.args.get('timestamp')
    csv_filename = f"{timestamp}.csv"
    if os.path.exists(csv_filename):
        results_df = pd.read_csv(csv_filename)
        return render_template('results.html', results=results_df.to_dict('records'))
    else:
        return "No results found."

if __name__ == '__main__':
    app.run(debug=True)