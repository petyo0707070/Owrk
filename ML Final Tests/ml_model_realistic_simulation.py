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

    # These will be used to identify events when they occur
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

    # This will hold the price at which we entered the position
    price_position = None

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
        if i < 20:
            pass


        else:
            # Computes the return of the trade price comparing the current trade price to the last one
            return_option = df.loc[i - 1, "price"]/ df.loc[i - 2, "price"] - 1

            # Compute the exponentially weighted volatility of the past 500 trades on the RETURNS
            last_500_trades = df[max(i - 500, 0): i]  # First need to get the last 500 trades including the current one
            last_500_trades.loc[:, "return"] = last_500_trades["price"] / last_500_trades["price"].shift(1) - 1
            sigma = last_500_trades["return"].ewm(span=500).std().values[-1]  # This is the current sigma


            last_500_trades["buy"] = last_500_trades["direction"].apply(lambda x: 1 if x == "buy" else 0)# This will be used later as a feature and in the "opposite_tick" exit approach

            # These rolling values are used to mark events i.e. an event is registered when they exceed the threshold
            sPos = max(0, sPos + return_option)
            sNeg = min(0, sNeg + return_option)

            # We will enter a position if an event occurs i.e. when sPos >= 2.5 * sigma or sNeg <= -2.5 * sigma,  TP and SL 10 * sigma * 0.5
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
                    int(i - 10), "price"]) / last_500_trades.loc[int(i - 10), "price"]
                features_list.append(return_last_10)

                # iv_last_trade
                iv_last_trade = last_500_trades.loc[int(i - 1), "iv"]
                features_list.append(iv_last_trade)

                # price_mark_price_deviation
                price_mark_price_deviation = (last_500_trades.loc[int(i - 1), "price"] - last_500_trades.loc[
                    int(i - 1), "mark_price"]) / last_500_trades.loc[int(i - 1), "price"]
                features_list.append(price_mark_price_deviation)

                # volume_last_10_trades
                volume_last_10_trades = last_500_trades.loc[int(i - 10): int(i - 1)]["contracts"].sum()
                features_list.append(volume_last_10_trades)

                # buy_last_10_trades
                buy_last_10_trades = last_500_trades.loc[int(i - 10): int(i - 1)]["buy"].sum()
                features_list.append(buy_last_10_trades)

                # index_return
                index_return = (last_500_trades.loc[i - 1, "index_price"] - last_500_trades.loc[i - 2, "index_price"]) /last_500_trades.loc[i - 2, "index_price"]
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

                """""
                Something very important is that we need to submit an order at the correct price that will be filled :), we need to guestimate what is the liquid price given the previous tick
                The logic is as follows:
                - Try to focus on OTM options, there the spread tends to be usually 0.001 BTC given that price is above 0.01
                - 0.005 to 0.01 is a gray area in terms of spread but again it is with a tick siye of 0.0005, so max spread 0.001 BTC
                - Option value below 0.005 is pretty sweet as the minimum tick is 0.0001
                - The idea is that if the side of the tick is opposite of the side which we enter, most likely we will be filled
                    but if the side of the trade is on the side of the position which we enter we need to account for the bid-ask spread
                - Add/Subtract 0.0002 to the tick if price below 0.005, Add/Subtract 0.0005 or 0.001 to the tick if price above 0.01
                - Do not enter if price is between 0.005 and 0.01
                """""

                print(f"Index is {i-1}, prediction is {prediction[0]}")
                print(X)

                # Go long if the prediction is 1 and the tick is not between 0.005 - 0.01
                if prediction[0] == 1 and (last_500_trades.loc[i-1, "price"] <= 0.005 or last_500_trades.loc[i-1, "price"] >= 0.01) :

                    # We start with the 2 cases whether the price is above 0.01 or belpw 0.005
                    position = 1
                    index_entered = i - 1

                    if last_500_trades.loc[i-1, "price"] <= 0.005:

                        # Here the tick on which the event was confirmed was the opposite side of the trade, so we get filled there
                        if last_500_trades.loc[i-1, "buy"] == 0:
                            price_position = last_500_trades.loc[i-1, "price"]

                        # Here the tick was in the direction of the position so we need to place the trade at a higher price to get filled
                        elif last_500_trades.loc[i-1, "buy"] == 1:
                          price_position = last_500_trades.loc[i-1, "price"] + 0.0002

                        # TP and SL are calculated based on the price at which we enteted !!!!!IMPORTANT!!!!!! might lead to inconsistencies
                        TP = price_position * (1 + 10 * sigma)
                        SL = price_position * (1 - 10 * sigma)

                    if last_500_trades.loc[i - 1, "price"] >= 0.01:

                        # Here the tick on which the event was confirmed was the opposite side of the trade, so we get filled there
                        if last_500_trades.loc[i - 1, "buy"] == 0:
                            price_position = last_500_trades.loc[i - 1, "price"]

                        # Here the tick was in the direction of the position so we need to place the trade at a higher price to get filled
                        elif last_500_trades.loc[i - 1, "buy"] == 1:
                            price_position = last_500_trades.loc[i - 1, "price"] + 0.001

                        # TP and SL are calculated based on the price at which we enteted !!!!!IMPORTANT!!!!!! might lead to inconsistencies
                        TP = price_position * (1 + 10 * sigma)
                        SL = price_position * (1 - 10 * sigma)


                # Go short if the prediction is -1 and the tick is not between 0.005 - 0.01
                if prediction[0] == -1 and (last_500_trades.loc[i-1, "price"] <= 0.005 or last_500_trades.loc[i-1, "price"] >= 0.01):
                    position = -1
                    index_entered = i - 1

                    if last_500_trades.loc[i - 1, "price"] <= 0.005:

                        # Here the tick on which the event was confirmed was the opposite side of the trade, so we get filled there
                        if last_500_trades.loc[i - 1, "buy"] == 1:
                            price_position = last_500_trades.loc[i - 1, "price"]

                        # Here the tick was in the direction of the position so we need to place the trade at a lower price to get filled
                        elif last_500_trades.loc[i - 1, "buy"] == 0:
                            price_position = last_500_trades.loc[i - 1, "price"] - 0.0002

                        # TP and SL are calculated based on the price at which we enteted !!!!!IMPORTANT!!!!!! might lead to inconsistencies
                        TP = price_position * (1 - 10 * sigma)
                        SL = price_position * (1 + 10 * sigma)

                    if last_500_trades.loc[i - 1, "price"] >= 0.01:

                        # Here the tick on which the event was confirmed was the opposite side of the trade, so we get filled there
                        if last_500_trades.loc[i - 1, "buy"] == 1:
                            price_position = last_500_trades.loc[i - 1, "price"]

                        # Here the tick was in the direction of the position so we need to place the trade at a higher price to get filled
                        elif last_500_trades.loc[i - 1, "buy"] == 0:
                            price_position = last_500_trades.loc[i - 1, "price"] - 0.001

                        # TP and SL are calculated based on the price at which we enteted !!!!!IMPORTANT!!!!!! might lead to inconsistencies
                        TP = price_position * (1 - 10 * sigma)
                        SL = price_position * (1 + 10 * sigma)

                # The fee to enter a trade is the smaller between 12.5% position size * option-price in USD or 0.03% * Index_price * position_size
                fee = min(0.125 * (last_500_trades.loc[i-1, "price"] * position_size * last_500_trades.loc[i-1, "index_price"]), 0.0003 * last_500_trades.loc[i-1, "index_price"] * position_size)

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
                    profit = (last_500_trades.loc[i-1, "price"] - price_position) * position_size * last_500_trades.loc[i - 1, "index_price"]

                    # Time to adjust the profit for the fees, which is very tricky
                    profit = profit - fee

                    # Append the trade to the array that holds them
                    trade_pnl.append(profit)

                    # This code snippet will update the equity curve
                    if len(equity_curve) == 0:
                        equity_curve.append(profit)
                    else:
                        equity_curve.append(profit + equity_curve[-1])


                    # Reset the trade specific attributes
                    fee = None
                    price_position = None
                    index_entered = None
                    TP = None
                    SL= None

                # This takes care if we are long and we hit SL
                if position == 1 and last_500_trades.loc[i-1, "price"] <= SL and last_500_trades.loc[i-1, "buy"] == 0:
                    position = 0 # Close the position

                    # We pay the fee again to exit the position
                    fee += min(0.125 * (last_500_trades.loc[i-1, "price"] * position_size * last_500_trades.loc[i-1, "index_price"]), 0.0003 * last_500_trades.loc[i-1, "index_price"] * position_size)

                    # Calculate the raw profit before fees and bid-ask spread
                    profit = (last_500_trades.loc[i-1, "price"] - price_position) * position_size * last_500_trades.loc[i - 1, "index_price"]

                    # Time to adjust the profit for the fees, which is very tricky
                    profit = profit - fee

                    trade_pnl.append(profit)

                    # This code snippet will update the equity curve
                    if len(equity_curve) == 0:
                        equity_curve.append(profit)
                    else:
                        equity_curve.append(profit + equity_curve[-1])


                    # Reset the trade specific attributes
                    fee = None
                    price_position = None
                    index_entered = None
                    TP = None
                    SL= None

                # This takes care if we are short and we hit TP
                if position == -1 and last_500_trades.loc[i - 1, "price"] <= TP and last_500_trades.loc[i - 1, "buy"] == 1:
                    position = 0  # Close the position

                    # We pay the fee again to exit the position
                    fee += min(0.125 * (last_500_trades.loc[i - 1, "price"] * position_size * last_500_trades.loc[i - 1, "index_price"]), 0.0003 * last_500_trades.loc[i - 1, "index_price"] * position_size)

                    # Calculate the raw profit before fees and bid-ask spread
                    profit = (price_position - last_500_trades.loc[i - 1, "price"]) * position_size * last_500_trades.loc[i - 1, "index_price"]

                    # Time to adjust the profit for the fees, which is very tricky
                    profit = profit - fee

                    trade_pnl.append(profit)

                    # This code snippet will update the equity curve
                    if len(equity_curve) == 0:
                        equity_curve.append(profit)
                    else:
                        equity_curve.append(profit + equity_curve[-1])

                    # Reset the trade specific attributes
                    fee = None
                    price_position = None
                    index_entered = None
                    TP = None
                    SL = None


                # This takes care if we are short and we hit SL
                if position == -1 and last_500_trades.loc[i - 1, "price"] >= SL and last_500_trades.loc[i - 1, "buy"] == 1:
                    position = 0  # Close the position

                    # We pay the fee again to exit the position
                    fee += min(0.125 * (last_500_trades.loc[i - 1, "price"] * position_size * last_500_trades.loc[i - 1, "index_price"]), 0.0003 * last_500_trades.loc[i - 1, "index_price"] * position_size)

                    # Calculate the raw profit before fees and bid-ask spread
                    profit = (price_position - last_500_trades.loc[i - 1, "price"]) * position_size * last_500_trades.loc[i - 1, "index_price"]

                    # Time to adjust the profit for the fees, which is very tricky
                    profit = profit - fee

                    trade_pnl.append(profit)

                    # This code snippet will update the equity curve
                    if len(equity_curve) == 0:
                        equity_curve.append(profit)
                    else:
                        equity_curve.append(profit + equity_curve[-1])

                    # Reset the trade specific attributes
                    fee = None
                    price_position = None
                    index_entered = None
                    TP = None
                    SL = None


    print(f"There were {len(events)}")
    print(trade_pnl)
    plt.plot(equity_curve)
    plt.show()


if __name__ == "__main__":
    main()