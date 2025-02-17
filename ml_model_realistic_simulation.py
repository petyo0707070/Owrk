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

    # This will keep track of the position size of options contract traded we will start with 0.1
    position_size = 0.1

    # This will be used to check against the exit so that we compute the return
    index_entered = None

    # This variable holds the ongoing fees associated with the current trade
    fee = None

    # Those will hold our PnL per trade and the evolution of our equity curve
    trade_pnl = []
    equity_curve = []

    # This will keep track of the SL and TP
    TP = None
    SL = None



    # This array will hold the indexes of the events which occured
    events = []

    """""""""
    This will determine the way positions are exited
    'opposite_tick' means that a position is exited when price goes outside TP/SL range and there is a tick in the opposite direction of the position
        i.e. you are long price goes above TP and your exit price is the first price traded where there is a sell trade that is above TP
    
    'generic' - exit on the price  that first trades at/over TP or SL
    """""""""
    exit_approach = "opposite_tick"

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
            return_option = (df.loc[i - 1, "price"] - df.loc[i - 2, "price"]) / df.loc[i - 1, "price"]

            # Compute the exponentially weighted volatility of the past 500 trades on the RETURNS
            last_500_trades = df[i - 500: i]  # First need to get the last 500 trades including the current one
            last_500_trades.loc[:, "return"] = last_500_trades["price"].diff() / last_500_trades["price"].shift(1)
            sigma = last_500_trades["return"].ewm(span=500).std().values[-1]  # This is the current sigma

            last_500_trades["buy"] = last_500_trades["direction"].apply(lambda x: 1 if x == "buy" else 0)# This will be used later as a feature and in the "opposite_tick" exit approach

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
                features_list.append(10 * sigma)

                # Return of the last 10 trades
                return_last_10 = (last_500_trades.loc[int(i - 1), "price"] - last_500_trades.loc[
                    int(i - 11), "price"]) / last_500_trades.loc[int(i - 11), "price"]
                features_list.append(return_last_10)

                # iv_last_trade
                iv_last_trade = last_500_trades.loc[int(i - 1), "iv"]
                features_list.append(iv_last_trade)

                # price_mark_price_deviation
                price_mark_price_deviation = (last_500_trades.loc[int(i - 1), "price"] - last_500_trades.loc[
                    int(i - 1), "mark_price"]) / last_500_trades.loc[int(i - 1), "price"]
                features_list.append(price_mark_price_deviation)

                # volume_last_10_trades
                volume_last_10_trades = last_500_trades.loc[int(i - 11): int(i - 1)]["contracts"].sum()
                features_list.append(volume_last_10_trades)

                # buy_last_10_trades
                buy_last_10_trades = last_500_trades.loc[int(i - 11): int(i - 1)]["buy"].sum()
                features_list.append(buy_last_10_trades)

                # index_return
                index_return = (last_500_trades.loc[i - 1, "index_price"] - last_500_trades.loc[i - 2, "index_price"]) / \
                               last_500_trades.loc[i - 2, "index_price"]
                features_list.append(index_return)

                # abnormal_trade_conditions
                abnormal_trade_conditions = np.where((pd.isna(last_500_trades.loc[i - 1, "block_trade_id"])) |
                                                     (pd.isna(last_500_trades.loc[i - 1, "combo_trade_id"])) |
                                                     (pd.isna(last_500_trades.loc[i - 1, "liquidation"])), True, False)
                features_list.append(abnormal_trade_conditions)

                # time_elapsed
                time_elapsed = last_500_trades.loc[i - 1, "timestamp"].timestamp() - last_500_trades.loc[
                    i - 2, "timestamp"].timestamp()
                features_list.append(time_elapsed)

                X = pd.DataFrame({"volatility": [features_list[0]],
                                  "ret_last_10": [features_list[1]],
                                  "iv_last_trade": [features_list[2]],
                                  "price_mark_price_deviation": [features_list[3]],
                                  "volume_last_10_trades": [features_list[4]],
                                  "buy_last_10_trades": [features_list[5]],
                                  "index_return": [features_list[6]],
                                  "abnormal_trade_conditions": [features_list[7]],
                                  "time_elapsed": [features_list[8]]})

                prediction = model.predict(X)

                # Go long if the prediction is 1
                if prediction[0] == 1:
                    position = 1
                    TP = df.loc[i, "price"] * (1 + 10 * sigma)
                    SL = df.loc[i, "price"] * (1 - 10 * sigma)
                    index_entered = i - 1

                # Go short if the prediction is -1
                if prediction[0] == -1:
                    position = -1
                    SL = df.loc[i, "price"] * (1 + 10 * sigma)
                    TP = df.loc[i, "price"] * (1 - 10 * sigma)
                    index_entered = i - 1

                # The fee to enter a trade is the smaller between 12.5% position size * option-price in USD or 0.03% * Index_price * position_size
                fee = min(0.125 * (last_500_trades.loc[i-1, "price"] * position_size * last_500_trades.loc[i-1, "index_price"]), 0.0003 * last_500_trades.loc[i-1, "index_price"] * position_size)
                print(fee)

            """""
            Something very important to consider is how the TP/SL are triggered.
            Imagine you are long there is a TP and SL, which type of tick provides a better approximation of reality
            Closing on the price of a sell tick that is above TP or below SL /favoured approach/, vice versa for shorts
            close on the buy tick below TP or above SL. The other options is to try to liquidate the position as soon as there is a tick outside the TP and SL range
            Refer to the variable exit_approach defined above
            """""
            # This part of the code is in charge of the trade management
            if exit_approach == "opposite_tick":

                # This takes care if we are long and we hit TP
                if position == 1 and last_500_trades.loc[i-1, "price"] >= TP and last_500_trades.loc[i-1, "buy"] == 0:
                    position = 0 # Close the position

                    # We pay the fee again to exit the position
                    fee += min(0.125 * (last_500_trades.loc[i-1, "price"] * position_size * last_500_trades.loc[i-1, "index_price"]), 0.0003 * last_500_trades.loc[i-1, "index_price"] * position_size)

                    # Calculate the raw profit before fees and bid-ask spread
                    profit = (last_500_trades.loc[i-1, "price"] - last_500_trades.loc[index_entered, "price"]) * position_size * last_500_trades.loc[i - 1, "index_price"]
                    # Time to adjust the profit for the fees, which is very tricky
                    trade_pnl.append(profit)




    print(f"There were {len(events)}")


if __name__ == "__main__":
    main()