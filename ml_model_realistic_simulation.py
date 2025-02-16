import pandas as pd
import numpy as np
import sys
import multiprocessing as mp
import datetime as dt
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as ss
from sklearn.metrics import precision_recall_fscore_support as score
import joblib
pd.options.mode.chained_assignment = None

def main():

    """""
    VERY IMPORTANT ASSUMPTIONS WE WILL MEASURE PNL IN USD not return because options are so volatile that simply returns will skew
    all the results quite a lot. ALSO BUY last_10 is actually the last 11 trades, everything that has last 10 is actually 11 :) somebody can't count
    """""

    model = joblib.load("tyranid_combined.pkl")

    # These will be used to idetify events when they occur
    sPos = 0 
    sNeg = 0

    # This will track if we are in position, 1 long, 0 None, -1 short
    position = 0

    # This will be used to check against the exit so that we compute the return
    index_entered = None

    # This will keep track of the SL and TP
    TP = None
    SL = None

    # This array will hold the indexes of the events which occured
    events = []

    # This is the option we are interested in trading
    df = pd.read_csv("btc_28_03_2025_call_105000.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(df)


    # Here is where the real unfolding will happen
    for i in df.index.values:

        # We do not do anything until the first 500 trades occur
        if i < 500:
            pass


        else:
            # Computes the return of the trade price comparing the current trade price to the last one
            return_option = (df.loc[i-1, "price"] - df.loc[i-2, "price"]) / df.loc[i-1,"price"]


            # Compute the exponentially weighted volatility of the past 500 trades on the RETURNS                
            last_500_trades = df[i- 500: i] # First need to get the last 500 trades including the current one
            last_500_trades.loc[:,"return"] = last_500_trades["price"].diff() / last_500_trades["price"].shift(1)
            sigma = last_500_trades["return"].ewm(span = 500).std().values[-1] # This is the current sigma

            # These rolling values are used to mark events i.e. an event is registered when they exceed the threshold
            sPos = max(0, sPos + return_option) 
            sNeg = min(0, sNeg + return_option)

            # We will enter a position if an event occurs i.e. when sPos >= 2.5 * sigma or sNeg <= -2.5 * sigma,  TP and SL 10 * sigma
            if position == 0 and (sPos >= 2.5 * sigma or sNeg <= -2.5 * sigma):

                # We need to reset them once they report an event
                if sPos >= 2.5 * sigma:
                    sPos = 0
                
                if sNeg <= - 2.5 * sigma:
                    sNeg = 0

                # Add the index of the event to the list
                events.append(i)

                # This will hold the features needed for the ML model
                features_list = []
                features_list.append(sigma)

                # Return of the last 10 trades
                return_last_10 = (last_500_trades.loc[int(i-1), "price"] - last_500_trades.loc[int(i - 11), "price"]) / last_500_trades.loc[int(i-11), "price"]
                features_list.append(return_last_10)

                # iv_last_trade
                iv_last_trade = last_500_trades.loc[int(i-1), "iv"]
                features_list.append(iv_last_trade)

                # price_mark_price_deviation
                price_mark_price_deviation = (last_500_trades.loc[int(i-1), "price"] - last_500_trades.loc[int(i-1), "mark_price"]) / last_500_trades.loc[int(i-1), "price"]
                features_list.append(price_mark_price_deviation)

                # volume_last_10_trades
                volume_last_10_trades = last_500_trades.loc[int(i-11): int(i-1)]["contracts"].sum()
                features_list.append(volume_last_10_trades)

                # buy_last_10_trades
                last_500_trades["buy"] = last_500_trades["direction"].apply(lambda x: 1 if x == "buy" else 0)
                buy_last_10_trades = last_500_trades.loc[int(i-11): int(i-1)]["buy"].sum()
                features_list.append(buy_last_10_trades)

                # index_return
                index_return = (last_500_trades.loc[i-1, "index_price"] - last_500_trades.loc[i-2, "index_price"])/last_500_trades.loc[i-2, "index_price"]
                features_list.append(index_return)

                #abnormal_trade_conditions
                abnormal_trade_conditions = np.where( (pd.isna(last_500_trades.loc[i - 1,"block_trade_id"])) |
                                                (pd.isna(last_500_trades.loc[i -1, "combo_trade_id"])) |
                                                (pd.isna(last_500_trades.loc[i-1, "liquidation"])), True, False)
                features_list.append(abnormal_trade_conditions)


                # time_elapsed
                print(last_500_trades.loc[i - 1, "timestamp"].timestamp())
                time_elapsed = last_500_trades.loc[i - 1, "timestamp"].timestamp() - last_500_trades.loc[i-2, "timestamp"].timestamp()
                features_list.append(time_elapsed)


                features_list.reshape(1, -1)
                prediction = model.predict(features_list)
                print(prediction)

                """""
                position = 1
                TP = df.loc[i, "price"] * (1 + 10 * sigma)
                SL = df.loc[i, "price"] * (1 - 10 * sigma)
                index_entered = 
                """
    print(f"There were {len(events)}")

if __name__ == "__main__":
    main()