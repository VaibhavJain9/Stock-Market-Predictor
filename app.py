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

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


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
          return "<span style='color: red; align: ; font-size: larger;'>Invalid stock symbol!</span>"

    start_date = request.form['start_date']
    end_date = request.form['end_date']

    stockprices = yf.download(symbol, start=start_date, end=end_date)

    test_ratio = 0.2
    training_ratio = 1 - test_ratio

    # Data Splitting
    train_size = int(training_ratio * len(stockprices))
    train = stockprices[:train_size][["Close"]]
    test = stockprices[train_size:][["Close"]]

    def extract_seqX_outcomeY(data, N, offset):
        X, y = [], []
        for i in range(offset, len(data)):
            X.append(data[i - N: i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def preprocess_test_data(data=stockprices, scaler=None, window_size=None, test=None):
        raw = data["Close"][len(data) - len(test) - window_size:].values
        raw = raw.reshape(-1, 1)
        raw = scaler.transform(raw)

        X_test = [raw[i - window_size: i, 0] for i in range(window_size, raw.shape[0])]
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stockprices[["Close"]])
    scaled_data_train = scaled_data[:train_size]

    # Preparing the LSTM model
    window_size = 50
    X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)

    def run_lstm_model(X_train, layer_units=100):
        inp = Input(shape=(X_train.shape[1], 1))
        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)
        out = Dense(1, activation="linear")(x)
        model = Model(inp, out)
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model

    model = run_lstm_model(X_train, layer_units=50)

    # Train the LSTM model
    cur_epochs = 5
    cur_batch_size = 32

    history = model.fit(
        X_train,
        y_train,
        epochs=cur_epochs,
        batch_size=cur_batch_size,
        verbose=1,
        validation_split=0.1,
        shuffle=True,
    )

    # Preprocess test data
    X_test = preprocess_test_data(data=stockprices, scaler=scaler, window_size=window_size, test=test)

    # Make predictions using the LSTM model
    predicted_price_ = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_)

    # Add the predicted prices to the test dataframe
    test["Predictions_lstm"] = predicted_price

    # Evaluate performance
    rmse_lstm = calculate_rmse(np.array(test["Close"]), np.array(test["Predictions_lstm"]))
    print("Root Mean Squared Error (RMSE) of LSTM model:", rmse_lstm)

    def plot_stock_trend_lstm(train, test):
        fig = plt.figure(figsize=(20, 10))
        plt.plot(train.index, train["Close"], label="Train Closing Price")
        plt.plot(test.index, test["Close"], label="Test Closing Price")
        plt.plot(test.index, test["Predictions_lstm"], label="Predicted Closing Price")
        plt.title(f"LSTM Model - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Stock Price ($)")
        plt.legend(loc="upper left")

        # Save the plot to a BytesIO object
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    # Return the plot using the defined function
    return plot_stock_trend_lstm(train, test)

if __name__ == '__main__':
    app.run(debug=True, port=5004)