import pandas as pd 
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Input
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys


# Function declarations
def stock_loss(y_true, y_pred):
    alpha = 100.0
    condition = tf.less(y_true * y_pred, 0)
    loss_true_case = alpha * y_pred**2 - tf.sign(y_true) * y_pred + tf.abs(y_true)
    loss_false_case = tf.abs(y_true - y_pred)
    loss = tf.where(condition, loss_true_case, loss_false_case)

    return tf.reduce_mean(loss, axis=-1)

def create_lstm_model():
    model = Sequential()
    
    model.add(Input(shape = (1,5)))
    # First LSTM layer with 4 neurons (input shape: 1 timestep, 4 features)
    model.add(LSTM(5, return_sequences=True, activation='tanh'))
    # Second LSTM layer with 3 neurons
    model.add(LSTM(3, return_sequences=True, activation='tanh'))
    # Third LSTM layer with 2 neurons
    model.add(LSTM(2, return_sequences = True,activation='tanh'))
    # Output layer with 1 neuron and tanh activation (output between -1 and 1)
    model.add(Dense(1, activation='tanh'))
    
    # Compile the model
    model.compile(optimizer='adam', loss= stock_loss)
    
    return model

def calculate_atr(df, period = 168):
    df_new = pd.DataFrame()
    df_new['high-low'] = df['high'] - df['low']
    df_new['high-close'] = abs(df['high'] - df['close'].shift(1))
    df_new['low-close'] = abs(df['low'] - df['close'].shift(1))
    df_new['tr'] = df_new[['high-low', 'high-close', 'low-close']].max(axis = 1)
    df_new['atr'] = df_new['tr'].rolling(window = period, min_periods = 1).mean()
    
    return df_new['atr']

def calculate_cmma(df, period):
    df_new = pd.DataFrame()
    df_new['cmma'] = (df['close'] - df['close'].rolling(window = period).mean())/df['atr'] * np.sqrt(lookback)

    normalizer = MinMaxScaler(feature_range=(-1,1))
    scaler = QuantileTransformer(output_distribution= 'normal')
    df_new['cmma'] = scaler.fit_transform(df_new[['cmma']])
    df_new['cmma'] = normalizer.fit_transform(df_new[['cmma']])
    return df_new['cmma']

def vsa_indicator(data: pd.DataFrame, norm_lookback: int = 168):
    # Norm lookback should be fairly large

    df_new = data
    df_new['atr'] = ta.atr(data['high'], data['low'], data['close'], norm_lookback)
    df_new['vol_med'] = data['volume'].rolling(norm_lookback).median()

    df_new['norm_range'] = (data['high'] - data['low']) / df_new['atr']
    df_new['norm_volume'] = data['volume'] / df_new['vol_med'] 

    norm_vol = df_new['norm_volume'].to_numpy()
    norm_range = df_new['norm_range'].to_numpy()

    range_dev = np.zeros(len(data))
    range_dev[:] = np.nan

    for i in range(norm_lookback * 2, len(data)):
        window = df_new.iloc[i - norm_lookback + 1: i+ 1]
        slope, intercept, r_val,_,_ = stats.linregress(window['norm_volume'], window['norm_range'])

        if slope <= 0.0 or r_val < 0.2:
            range_dev[i] = 0.0
            continue
       
        pred_range = intercept + slope * norm_vol[i]
        range_dev[i] = norm_range[i] - pred_range
    
    df_new['vsa'] = range_dev
    return df_new['vsa']


# Loading the Data
df = pd.read_csv('trading_view_spy_1_and_2_converted.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
test_df = df[13260:]
df = df[0:13250]
lookback = len(df)


# Calculating the inputs and the outputs
df['atr'] = calculate_atr(df)
df['cmma_6'] = calculate_cmma(df, 6)
df['cmma_24'] = calculate_cmma(df, 24)
df['cmma_72'] = calculate_cmma(df, 72)
df['vsa'] = vsa_indicator(df)
df['Log Return'] = np.log(df['close'].shift(-1)/df['close'])
df = df.dropna()
lookback = len(df)



# Prepare the input for the model
cmma_6 = df['cmma_6'].values
cmma_24 = df['cmma_24'].values
cmma_72 = df['cmma_72'].values
vsa = df['vsa'].values
cmma_6 = cmma_6.reshape((cmma_6.shape[0], 1, 1))
cmma_24 = cmma_24.reshape((cmma_24.shape[0], 1, 1))
cmma_72 = cmma_72.reshape((cmma_72.shape[0], 1, 1))
vsa = vsa.reshape((cmma_72.shape[0], 1, 1))
model_output = np.zeros((cmma_6.shape[0], 1,1))
X_train = np.column_stack((cmma_6, cmma_24, cmma_72, vsa, model_output))
X_train = X_train.reshape((lookback,1 ,5))
Y_train = np.array(df['Log Return'].values)


# Train the first itteraton of the model
model = create_lstm_model()
model.fit(X_train, Y_train, epochs = 10, batch_size = 2)
model_output = (model.predict(X_train)).reshape((cmma_6.shape[0], 1, 1))

#Train the second itteration of the model
X_train = np.column_stack((cmma_6, cmma_24, cmma_72, vsa, model_output))
X_train = X_train.reshape((lookback,1 ,5))
model.fit(X_train, Y_train, epochs = 10, batch_size = 2)

# Some variables needed for the out of sample results
returns = [0]
previous_output = 0
first = 1


# Prepare the input for the testing stage
test_df['atr'] = calculate_atr(test_df)
test_df['cmma_6'] = calculate_cmma(test_df, 6)
test_df['cmma_24'] = calculate_cmma(test_df, 24)
test_df['cmma_72'] = calculate_cmma(test_df, 72)
test_df['vsa'] = vsa_indicator(test_df)
test_df['Log Return'] = np.log(test_df['close'].shift(-1)/test_df['close'])
test_df = test_df.dropna()



for i in range(250):
    if i % 100 ==0:
        print(i)
    

    cmma_data = test_df.iloc[i][['cmma_6', 'cmma_24', 'cmma_72', 'vsa']].values
    x_data = np.append(cmma_data, previous_output)
    x_data = x_data.reshape(1, 1, 5)
    output = model.predict(x_data)[0][0]
    previous_output = output
    result = test_df.iloc[i]['Log Return'] * output
    previous_return = returns[-1]
    returns.append(previous_return + result[0])

plt.plot(returns)
plt.show()


    
