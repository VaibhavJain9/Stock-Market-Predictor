import io
from flask import Response, Flask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')
from flask import render_template, request,jsonify




app = Flask(__name__)




def is_valid_symbol(symbol):
    try:
        data = yf.download(symbol, period="1d")  # Download 1 day of data
        if data.empty:
            return False
        else:
            return True
    except:
        return False
    
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-chart', methods=['POST'])
def generate_chart():
    symbol = request.form['symbol']
    if not is_valid_symbol(symbol):
         return render_template('inval.html')

    start_date = request.form['start_date']
    end_date = request.form['end_date']

    stockprices = yf.download(symbol, start=start_date, end=end_date)

    # Data Preparation
    train = stockprices[["Open", "High", "Low", "Volume", "Close"]]
    
    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(train)
    
    def extract_seqX_outcomeY(data, N, offset):
        X, y = [], []
        for i in range(offset, len(data)):
            X.append(data[i - N: i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    window_size = 50
    X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
    
    def run_lstm_model(X_train, layer_units=30):
        inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)
        out = Dense(5, activation="linear")(x)  # Output shape matches number of features
        model = Model(inp, out)
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model
    
    model = run_lstm_model(X_train, layer_units=5)
    
    # Train the LSTM model
    cur_epochs = 20
    cur_batch_size = 32
    
    history = model.fit(
        X_train,
        y_train,
        epochs=cur_epochs,
        batch_size=cur_batch_size,
        verbose=1,
        shuffle=True,
    )
    
    # Prepare input for forecasting
    last_sequence = X_train[-1]
    forecasted_prices = []
    forecast_length = 30
    for _ in range(forecast_length):
        predicted_values = model.predict(np.array([last_sequence]))[0]
        forecasted_prices.append(predicted_values)
        last_sequence = np.vstack((last_sequence[1:], np.array([predicted_values])))
    
    # Convert forecasted prices back to original scale
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices))
    
    # Extend the date index for plotting the forecast
    future_dates = pd.date_range(start=stockprices.index[-1], periods=forecast_length + 1)
    
    # Plot the forecasted prices along with actual closing prices
    def plot_stock_trend_lstm(train, forecasted_prices, future_dates):
        fig = plt.figure(figsize=(20, 10))
        plt.plot(train.index, train["Close"], label="Actual Closing Price")
        plt.plot(future_dates[1:], forecasted_prices[:, 4], label="Forecasted Closing Price (Next 30 Days)")
        plt.title(f"LSTM Model - {symbol} with Forecast (Multivariate)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend(loc="upper left")
        plt.show()
    
   
        # Save the plot to a BytesIO object
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    # Return the plot using the defined function
    return plot_stock_trend_lstm(train, forecasted_prices, future_dates)

if __name__ == '__main__':
    app.run(debug=True, port=5004)