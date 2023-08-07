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
from tensorflow.keras.models import Sequential
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

    stock_prices = yf.download(symbol, start=start_date, end=end_date)
    stock_prices_df = pd.DataFrame(stock_prices)

    
    cols = list(stock_prices_df)[1:7]
    
    df_for_training = stock_prices_df[cols].astype(float)
    
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    
    trainX = []
    trainY = []
    
    n_future = 1  
    n_past = 14  
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
     trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
     trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY) 
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    
    model.add(Dense(trainY.shape[1]))
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
    
    n_past = 16
    n_days_for_prediction = 30
    
    predict_period_dates = pd.date_range(list(stock_prices_df.index)[-n_past], periods=n_days_for_prediction, freq='B').tolist()
    
    prediction = model.predict(trainX[-n_days_for_prediction:])
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
    
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
    
    df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    
    # Filter df_forecast to include only rows with dates greater than or equal to end_date
    df_forecast_filtered = df_forecast[df_forecast['Date'] >= end_date]
    
    fig, ax = plt.subplots()
    
   
    ax.plot(stock_prices_df.index, stock_prices_df['Open'], label='Original')
    ax.plot(df_forecast_filtered['Date'], df_forecast_filtered['Open'], label='Predicted')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Open Price')
    ax.set_title('Stock Price Prediction')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

    
    # Save the plot to a BytesIO object
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
    # Return the plot using the defined function
    return plot_stock_trend_lstm(train, test)
    

if __name__ == '__main__':
    app.run(debug=True, port=5004)