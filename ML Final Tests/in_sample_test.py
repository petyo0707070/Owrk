import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
import sklearn.metrics
import sklearn.model_selection
from trendline_break_dataset import trendline_breakout_dataset
from treandline_beak_dataset_support import trendline_breakout_dataset_support
from trendline_automation import fit_trendlines_single
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import sklearn

def plot_trade(ohlc: pd.DataFrame, trades: pd.DataFrame, trade_i: int, lookback: int):
    plt.style.use('dark_background')

    trade = trades.iloc[trade_i]
    entry_i = int(trade['entry_i'])
    exit_i = int(trade['exit_i']) 
    
    candles = np.log(ohlc.iloc[entry_i - lookback:exit_i+1])
    resist = [(candles.index[0], trade['intercept_r']), (candles.index[lookback], trade['intercept_r'] + trade['slope_r'] * lookback)]
    support = [(candles.index[0], trade['intercept_s']), (candles.index[lookback], trade['intercept_s'] + trade['slope_s'] * lookback)]
    tp = [(candles.index[lookback], trade['tp']), (candles.index[-1], trade['tp'])]
    sl = [(candles.index[lookback], trade['sl']), (candles.index[-1], trade['sl'])]

    mco = [None] * len(candles)
    mco[lookback] = 'blue'
    fig, axs = plt.subplots(2, sharex=True, height_ratios=[3, 1])
    axs[1].set_title('Volume')

    mpf.plot(candles, volume=axs[1], alines=dict(alines=[resist, tp, sl, support], colors=['w', 'b', 'r', 'w']), type='candle', style='charles', ax=axs[0], marketcolor_overrides=mco)
    mpf.show()



# Own functions for plotting equity curves

def plot_equity_curve_test(y_test_new, y_pred, trades):
    itt = -1
    total_number_trades = 0
    winning_trades = 0

    return_array = []

    for element in y_pred:
        itt += 1
        if element == 1:
            total_number_trades += 1
            y_test_value = y_test_new.iloc[itt,]
            if element == y_test_value:
                winning_trades += 1
                return_array.append(abs(trades.loc[int(len(y_train)) + itt, "return"]))
                #print(f"itt is {itt}, len is {len(y_train)}, len + itt is {itt + len(y_train)}")
                #print(y_test_new.iloc[itt,])
                #print(trades.loc[int(len(y_train) + itt),])
            else:
                return_array.append(- abs(trades.loc[int(len(y_train)) + itt, "return"]))
    
    equity_curve = [1]

    for i in range(0 , len(return_array)):
        print(i)
        equity_curve.append(equity_curve[i] * (1 + return_array[i]))

    print(f"There were {total_number_trades} trades, out of which {winning_trades} were winning trades")
    accuracy = sklearn.metrics.accuracy_score(y_test_new, y_pred)
    precision = sklearn.metrics.precision_score(y_test_new, y_pred)
    print(f"Accuracy in test is {accuracy}, precision is {precision}")

    plt.plot(equity_curve)
    plt.title("Equity Curve Testing")
    plt.show()

def plot_equity_curve_train(y_train_new, y_pred, trades):
    itt = -1
    total_number_trades = 0
    winning_trades = 0

    return_array = []

    for element in y_pred:
        itt += 1
        if element == 1:
            total_number_trades += 1
            y_train_value = y_train_new.iloc[itt,]
            if element == y_train_value:
                winning_trades += 1
                return_array.append(abs(trades.loc[int(itt), "return"]))
                #print(f"itt is {itt}, len is {len(y_train)}, len + itt is {itt + len(y_train)}")
                #print(y_test_new.iloc[itt,])
                #print(trades.loc[int(len(y_train) + itt),])
            else:
                return_array.append(- abs(trades.loc[int(itt), "return"]))
    
    equity_curve = [1]

    for i in range(0 , len(return_array)):
        print(i)
        equity_curve.append(equity_curve[i] * (1 + return_array[i]))


    print(f"There were {total_number_trades} trades, out of which {winning_trades} were winning trades")
    accuracy_train = sklearn.metrics.accuracy_score(y_train_new, y_pred )
    precision_train = sklearn.metrics.precision_score(y_train_new, y_pred )
    print(f"Accuracy in the  train dataset is {accuracy_train}, precision is {precision_train}")


    plt.plot(equity_curve)
    plt.title("Equity Curve Training")
    plt.show()




if __name__ == "__main__":
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()
    data = data[(data.index > '2018-01-01') & (data.index < '2020-12-31')]

    plt.style.use('dark_background')


    #trades, data_x, data_y = trendline_breakout_dataset(data, 72, hold_period = 24)
    trades, data_x, data_y = trendline_breakout_dataset_support(data, 72, hold_period = 24)

    print(trades)


    # Set up Random Forest Bagging CLassifier to identify extreme events 1 if the events is extrame, 0 if not
    trades["label_size"] = np.where(trades["return"]  < -0.03, 1, 0)

    # Add trade dependency
    trades["previous_return"] = trades["return"].shift(1)


    #trades["label_size"] = np.where(trades["return"] > 0.03, 1, np.where(trades['return'] < - 0.03, 1, 0))
    #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(trades[['resist_s', "support_s" ,'tl_err_r', "tl_err_s",'vol', 'max_dist_r', 'max_dist_s']], trades["label_size"],test_size= 0.36, shuffle= False)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(trades[["support_s" , "tl_err_s",'vol',  'max_dist_s']], trades["label_size"],test_size= 0.36, shuffle= False)



    """""
     Define the model things to consider !!!!!!:
     - criterion : "gini"?
     - max_depth : 3 might be good
     - min_samples_leaf: default or not
    """
    model = RandomForestClassifier(n_estimators = 100, max_depth= 4, criterion="gini", class_weight= "balanced_subsample", bootstrap= False, min_samples_leaf = 0.05)
    model = BaggingClassifier(estimator= model, n_estimators= 100)
    model.fit(X_train, y_train)


    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    
    # We are not looking at precision but rather at the trades where we got a prediction of 1 whether they resulted in
    trades["return_label"] = np.where(trades["return"] < -0.001 , 1, 0)
    X_train, X_test, y_train_new, y_test_new = sklearn.model_selection.train_test_split(trades[["support_s" , "tl_err_s",'vol',  'max_dist_s']], trades["return_label"],test_size= 0.36, shuffle= False)


    plot_equity_curve_train(y_train_new, y_pred_train, trades)
    plot_equity_curve_test(y_test_new, y_pred_test, trades)



    #plt.hist(trades["return"], bins = 30)
    #plt.show()


    for i in range(0, len(trades)):
        plot_trade(data, trades, i, 72)
    trades.plot.scatter('resist_s', 'return')
    plt.show()


