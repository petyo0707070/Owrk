import datetime

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as ss
from sklearn.metrics import precision_recall_fscore_support as score


def main(df, differentiate_bool=False, eth=False):

    # Abnormal trade conditions are defined as anything that is either a liquidation, block or combo trade
    df["abnormal_trade_conditions"] = np.where( (df["block_trade_id"].isnull()) |
                                                (df["combo_trade_id"].isnull()), True, False)# |
                                                #(df["liquidation"].isnull()), True, False)



    # Ensuring that timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # The time since last trade in seconds
    df["time_elapsed"] = df["timestamp"].diff().dt.total_seconds()


    df["direction"] = df["direction"].apply(lambda x: 1 if x == 'buy' else 0)
    df["index_return"] = df["index_price"].diff() / df["index_price"].shift(1)

    # Get the time to expiration
    df["time_to_expiration"] = get_time_to_expiration(df)


    # Calculate how far away the underlying price is from the option strike price in %, positive values OTM, negative ITM
    df["strike_underlying_distance"] = get_moneyness(df)

    #Get the Hawkes Process with volume imbalance as a feature, higher values of kappa mean faster decay, discounted by t measured in days
    df["hawkes"] = hawkes_process(df["contracts"] * df["direction"].apply(lambda x: 1 if x == 1 else -1), df["time_elapsed"], 0.5)

    # Get an estimator for the linearity of price
    df["PTSR"] = rw_ptsr(df["price"].to_numpy(), 200)

    print(df)

    if differentiate_bool:
        differentiator = plotMinFFD(df)
        stationary_series = fracDiff_FFD(np.log(df[['close']]), differentiator, 1e-4 / 2)
        stationary_series = np.exp(stationary_series)
        print(stationary_series)
    else:
        stationary_series = df[['price']]

    # Calculate the exponentially weighted volatility of the past 100 trades
    df0 = stationary_series / stationary_series.shift(1) - 1  # returns
    sigma = df0.ewm(span=200).std()

    # We drop the na values and select the rows from the stationary series which are the ones
    # which did not get dropped
    sigma = sigma.dropna()
    stationary_series = stationary_series[stationary_series.index.isin(sigma.index)]

    events = getTEvents(stationary_series['price'], 10 * sigma)

    t1 = stationary_series['price'].index.searchsorted(events + 100)
    t1 = t1[t1 < stationary_series['price'].shape[0]]
    t1 = pd.Series(stationary_series['price'].index[t1], index=events[:t1.shape[0]])


    df1 = getEvents(stationary_series['price'], events, [1 , 1], 10 * sigma, 0.001, 48, t1, None, eth=eth, raw_data=df)
    df1 = df1.sort_values(by='start', ascending=True)

    """""
    What needs to be done is to sort by start and track the events chronologically instead of sorting by end and dropping things that is very important !!!!
    Something else also very urgent to consider is the sigma threshold for sPos and sNeg as it stands right now 2.5 is very little, roughly 10% of ticks get flagged as an event !!!
    Also returns on the option are not representative of dollar PNL so might be better evaluate performance with dollar PNL
    """""


    # For now we will go off the assumption starting with that the training will include all the trades while validation and testing only the chronological ondes
    #df1 = get_chronological_events(df1)

    df1['iv_last_trade'] = df1["start"].apply(lambda x: df.loc[x, "iv"])
    df1["volatility"] = df1["start"].apply(lambda x: sigma.loc[x,])
    df1['buy'] = df1["start"].apply(lambda x: df.loc[x, "direction"])
    df1['price_mark_price_deviation'] = df1["start"].apply(
        lambda x: (df.loc[x, "price"] - df.loc[x, 'mark_price']) / df.loc[x, 'price'])
    df1['volume_last_10_trades'] = df1["start"].apply(lambda x: df['contracts'].loc[float(int(x) - 9): x].sum())
    df1["tick_direction"] = df1["start"].apply(lambda x: df.loc[x, "tick_direction"])
    df1['buy_last_10_trades'] = df1["start"].apply(lambda x: df['direction'].loc[float(int(x) - 9): x].sum())
    df1["index_return"] = df1["start"].apply(lambda x: df.loc[x, "index_return"])
    df1["abnormal_trade_conditions"] = df1["start"].apply(lambda x: df.loc[x, "abnormal_trade_conditions"])
    df1["time_elapsed"] = df1["start"].apply(lambda x: df.loc[x, "time_elapsed"])
    df1["time_to_expiration"] = df1["start"].apply(lambda x: df.loc[x, "time_to_expiration"])
    df1["strike_underlying_distance"] = df1["start"].apply(lambda x: df.loc[x, "strike_underlying_distance"])
    df1["hawkes"] = df1["start"].apply(lambda x: df.loc[x, "hawkes"])
    df1["PTSR"] = df1["start"].apply(lambda x: df.loc[x, "PTSR"])
    # Drop events in the warmup period but START COUNTING SPOS AND SNEG AS SOON AS THE FIRST TRADE
    df1.dropna(inplace = True)

    print(df1)


    #features_list = df1[["volatility", "ret_last_10", "iv_last_trade", "price_mark_price_deviation", "volume_last_10_trades", "buy_last_10_trades", "index_return", "time_elapsed", "time_to_expiration", "strike_underlying_distance", "abnormal_trade_conditions", "hawkes", "PTSR" ]]
    features = ["PTSR", "hawkes", "strike_underlying_distance", "time_to_expiration"]

    features_list = df1[features]

    print(df1.groupby("bin").size())

    #best_performer_per_feature = generate_model_proposals(model = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False,
    #                                       class_weight='balanced_subsample',
    #                                       min_weight_fraction_leaf=0.05, n_jobs = 1),
    #                                        X = features_list[0 : int(0.64 * len(features_list))],
    #                                        y = df1["bin"][0: int(0.64 * len(df1))],
    #                                        max_n_features = 5)

    #print(best_performer_per_feature)



    # This implements PCA / Principal Component Analysis/ even before we train the model in an attempt to make it less overfit

    #X_train, X_validation_test, y_train, y_validation_test = train_test_split(df1[["volatility", 'ret last 5', 'iv_last_trade', 'buy',
    #                                                         'price_mark_price_deviation', 'volume_last_10_trades']],
    #                                                    df1['bin'], test_size=0.36, shuffle=False)

    X_train, X_validation_test, y_train, y_validation_test = train_test_split(features_list, df1['bin'], test_size=0.36, shuffle=False)




    # This part here insures that we only get the trades we would have entered in the validation and test set ?????
    # Perhaps it is better to include all in order to train the model on as many events as possible
    df_temp = df1[df1.index.isin(X_validation_test.index)]
    df_temp = get_chronological_events(df_temp)
    X_validation_test = df_temp[features]
    y_validation_test = df_temp["bin"].astype(float)


    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size = 0.5, shuffle= False)
    #append_to_combined_data(X_train, X_validation, X_test, y_train, y_validation, y_test)


    # orthogonal_features_train = orthoFeats(X_train)
    # X_train = pd.DataFrame(orthogonal_features_train, columns=['volatility', 'ret last 5', 'bullish last 5'])

    # orthogonal_features_test = continiousOrthoFeats(df1[['volatility', 'ret last 5', 'bullish last 5']])
    # X_test = pd.DataFrame(orthogonal_features_test, columns=['volatility', 'ret last 5', 'bullish last 5']).loc[X_test.index]

    indMatrix = getIndMatrix(df.index, t1.iloc[0:len(X_train)])
    avgU = getAvgUniqueness(indMatrix).sum() / len(getAvgUniqueness(indMatrix))

    def bagging_clasifier():
        baggin_classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            oob_score=False
        )

        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=False)
        cross_score = cross_val_score(baggin_classifier, X_train, y_train, cv=stratified_kfold, scoring="f1")
        print(f"Mean cross-validation score: {np.mean(cross_score)}")
        baggin_classifier.fit(X_train, y_train)
        y_pred = baggin_classifier.predict(X_test)

    # 3 Ways to set up random forests
    def random_forest(i=2, avgU=avgU, X_train = X_train, y_train = y_train):

        if i == 0:
            model = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', criterion='entropy')

        elif i == 1:
            model = DecisionTreeClassifier(criterion='entropy', max_features=0.75, class_weight='balanced')
            model = BaggingClassifier(estimator=model, n_estimators=1000, max_samples=avgU)

        elif i == 2:
            model = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True,
                                           class_weight='balanced_subsample',
                                           min_weight_fraction_leaf=0.05, n_jobs = 1, max_depth=3)
            model = BaggingClassifier(estimator=model, n_estimators = 100, max_samples=avgU)  # , max_features=1.)

            walk_forward_precision, walk_forward_accuracy = evaluate_model(model, X_train, y_train)
            print(f"The precision of the walk forward is {walk_forward_precision}")
            print(f"The accuracy of the walk forward is {walk_forward_accuracy}")



        # stratified_kfold = StratifiedKFold(n_splits=10, shuffle=False)
        # cross_score = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring="f1")
        # print(f"Cross-validation score for the random-forest model is {np.mean(cross_score)}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation)

        df2_training, df2_validation = simple_stats(y_pred, model = model)

        if differentiate_bool == True:
            results_from_raw_data(df2_validation, df)

        start = df1.loc[0, 't1']
        end = df1.loc[len(X_train), 't1']
        frequency = (end - start).days / 252
        prob_failure = probFailure(X_train['ret last 5'], frequency, 1)
        print(f"The chance the strategy fails only from the training data is {prob_failure}")

        start = df1.loc[len(X_train), 't1']
        end = df1['t1'].iloc[-1]
        frequency = (end - start).days / 252
        prob_failure = probFailure(X_test['ret last 5'], frequency, 1)

        print(f"The chance the strategy fails only from the testing data is {prob_failure}")

        sys.exit()



    def simple_stats(y_pred, model,df1=df1):
        from sklearn.metrics import accuracy_score

        # This shows the performance of our model on the training dataset
        y_pred_train = model.predict(X_train)
        result_training = y_pred_train == y_train
        precision_train, recall_train, fscore_train, support_train= score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        print(f"Precision for the training set is {precision_train}, Accuracy is {accuracy_train}")
        df_train = df1.loc[y_train.index]
        df_train = df_train.reset_index(drop = True)
        df_train["bool"] = result_training
        df_train["prediction"] = y_pred_train
        df_train["result"] = np.where(df_train['prediction'] == 1, (df_train['USD PNL']),
                                 np.where(df_train['prediction'] == - 1, ((df_train['USD PNL'] * - 1)), 1))
        df_train['result'] = df_train['result'].cumsum()

        df_train['result'].plot(kind='line', title='USD PNL of the RF Bagged Model on the training set')
        plt.show()

        #sys.exit()




        # This shows the performance of our model on the validation dataset
        y_pred_validation = model.predict(X_validation)
        result_validation = y_pred_validation == y_validation

        precision_validation, recall_validation, fscore_validation, support_validation = score(y_validation, y_pred_validation)
        accuracy_validation = accuracy_score(y_validation, y_pred_validation)
        print(f"Precision for the validation set is {precision_validation}, Accuracy is {accuracy_validation}")
        df_validation = df1.loc[y_validation.index]
        df_validation = df_validation.reset_index( drop = True)
        df_validation["bool"] = result_validation
        df_validation["prediction"] = y_pred_validation
        df_validation["result"] = np.where(df_validation['prediction'] == 1, (df_validation['USD PNL']),
                                      np.where(df_validation['prediction'] == - 1, ((df_validation['USD PNL'] * - 1)), 1))
        df_validation['result'] = df_validation['result'].cumsum()
        df_validation['result'].plot(kind='line', title='USD PNL of the RF Bagged Model on the validation set')
        plt.show()

        sys.exit()



        # This shows the performance of our model on the test dataset
        y_pred_test = model.predict(X_test)
        result_test = y_pred_test == y_test

        precision_test, recall_test, fscore_test, support_test = score(y_test, y_pred_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print(f"Precision for the test set is {precision_test}, Accuracy is {accuracy_test}")
        df_test = df1.loc[y_test.index]


        df_test = df1.loc[y_test.index]
        df_test["bool"] = result_test
        df_test["prediction"] = y_pred_test
        df_test["result"] = np.where(df_test['prediction'] == 1, (df_test['USD PNL']),
                                      np.where(df_test['prediction'] == - 1, ((df_test['USD PNL'] * - 1)), 1))
        df_test['result'] = df_test['result'].cumsum()
        df_test['result'].plot(kind='line', title='USD PNL of the RF Bagged Model on the test set')
        plt.show()


        save_model(model)

        sys.exit()

        return df_train, df_validation, df_test

    random_forest(2)




def append_to_combined_data(X_train, X_validation, X_test, y_train, y_validation, y_test):

    # Get the existing data
    x_train_existing = pd.read_csv("x_train.csv")
    x_validation_existing =pd.read_csv("x_validation.csv")
    x_test_existing = pd.read_csv("x_test.csv")

    y_train_existing = pd.read_csv("y_train.csv")
    y_validation_existing = pd.read_csv("y_validation.csv")
    y_test_existing = pd.read_csv("y_test.csv")

    print("Data successfully appended")



    # Append the current information for df1 to the existing big dataset
    x_train_new = pd.concat([x_train_existing, X_train])
    x_validation_new = pd.concat([x_validation_existing, X_validation])
    x_test_new = pd.concat([x_test_existing, X_test])

    y_train_new = pd.concat([y_train_existing, y_train])
    y_validation_new = pd.concat([y_validation_existing, y_validation])
    y_test_new = pd.concat([y_test_existing, y_test])

    x_train_new.to_csv("x_train.csv", index = False)
    y_train_new.to_csv("y_train.csv", index = False)
    x_validation_new.to_csv("x_validation.csv", index = False)
    y_validation_new.to_csv("y_validation.csv", index = False)
    x_test_new.to_csv("x_test.csv", index = False)
    y_test_new.to_csv("y_test.csv", index = False)


# Implement ordinal patterns permutation as a proxy of the linearity of pricce !!! Low Linearity better mean reversion, dimensiotality is set at 3
# Only hyper parameter needed is lookback and the close array needs to be in a numpy format
def ordinal_patterns(arr: np.array, d: int) -> np.array:
    import math
    assert (d >= 2)
    fac = math.factorial(d);
    d1 = d - 1
    mults = []
    for i in range(1, d):
        mult = fac / math.factorial(i + 1)
        mults.append(mult)

    # Create array to put ordinal pattern in
    ordinals = np.empty(len(arr))
    ordinals[:] = np.nan

    for i in range(d1, len(arr)):
        dat = arr[i - d1:  i + 1]
        pattern_ordinal = 0
        for l in range(1, d):
            count = 0
            for r in range(l):
                if dat[d1 - l] >= dat[d1 - r]:
                    count += 1

            pattern_ordinal += count * mults[l - 1]
        ordinals[i] = int(pattern_ordinal)

    return ordinals


def perm_ts_reversibility(arr: np.array):
    import scipy
    # Zanin, M.; Rodríguez-González, A.; Menasalvas Ruiz, E.; Papo, D. Assessing time series reversibility through permutation

    # Should be fairly large array, very least ~60
    assert (len(arr) >= 10)
    rev_arr = np.flip(arr)

    # [2:] drops 2 nan values off start of val
    pats = ordinal_patterns(arr, 3)[2:].astype(int)
    r_pats = ordinal_patterns(rev_arr, 3)[2:].astype(int)

    # pdf of patterns, forward and reverse time
    n = len(arr) - 2
    p_f = np.bincount(pats, minlength=6) / n
    p_r = np.bincount(r_pats, minlength=6) / n

    if min(np.min(p_f), np.min(p_r)) > 0.0:
        rev = scipy.special.rel_entr(p_f, p_r).sum()
    else:
        rev = np.nan

    return rev


def rw_ptsr(arr: np.array, lookback: int):
    # Rolling window permutation time series reversibility
    rev = np.zeros(len(arr))
    rev[:] = np.nan

    lookback_ = lookback + 2
    for i in range(lookback_, len(arr)):
        dat = arr[i - lookback_ + 1: i + 1]
        rev_w = perm_ts_reversibility(dat)

        if np.isnan(rev_w):
            rev[i] = rev[i - 1]
        else:
            rev[i] = rev_w

    return rev


def get_moneyness(df):
    # Calculate how far away the underlying price is from the option strike price in %, positive values OTM, negative ITM
    underlying_option = df.loc[0, "instrument_name"]
    strike = int(underlying_option[12:-2])
    df["moneyness"] = (strike - df["index_price"])/df["index_price"]
    return df["moneyness"]


# timedelta is measured in seconds since last trade
def hawkes_process(data: pd.Series, time_delta ,kappa: float):
    assert(kappa > 0.0)
    time_delta = time_delta.to_numpy()
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            time_elapsed_days = time_delta[i] / 86400
            alpha = np.exp(-kappa * time_elapsed_days)
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def get_time_to_expiration(df):
    month_dict = {"JAN": 1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6, "JUL":7, "AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12}

    underlying_option = df.loc[0, "instrument_name"]
    year = int("20" + underlying_option[9:11])
    month = month_dict[underlying_option[6:9]]
    day = int(underlying_option[4:6])

    expiration_date = datetime.datetime(year, month, day, 9, 0, 0)
    df["time_to_expiration"] = (expiration_date - df["timestamp"]).dt.total_seconds()
    return df["time_to_expiration"]

# Evaluates a model using a walk-forward test
def evaluate_model(model, X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model.fit(X, y)


    cv = StratifiedKFold(n_splits = 5)
    score_precision = cross_val_score(model, X, y, scoring = "precision_weighted", cv = cv, n_jobs = 16, error_score = "raise")
    score_accuracy = cross_val_score(model, X, y, scoring = "balanced_accuracy", cv = cv, n_jobs = 16, error_score="raise")

    return score_precision, score_accuracy

def generate_model_proposals(model, X, y, max_n_features = 5):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline

    model_dict = dict()

    model_result_dict = dict()

    # This picks models based on numbers of features
    for i in range(2, max_n_features + 1):
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select= i)
        model_dict[str(i)] = Pipeline(steps = [("s", rfe), ("m", model)])


   # This evaluates every optimal model for a given n features
    for name, model in model_dict.items():

        score = evaluate_model(model, X, y)
        model_result_dict[name] = score


    return model_result_dict


def save_model(model):
    import joblib
    joblib.dump(model, "tyranid_fitted_light.pkl")


def plot_correlation_matrix(df):
    import seaborn as sns
    corr = df.corr(method = "kendall")
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, fmt = ".2f", cmap = "coolwarm")
    plt.show()


# This gets the events in a chronological order with the idea that we will be in only one trade at a time
def get_chronological_events(df):
    df = df.sort_values(by='start', ascending=True)

    #Create an a dataframe with the same columns and data types as the original, but it is empty
    df1 = df.iloc[:0,:].copy()

    end_current_trade = None
    first = 1

    for index, row in df.iterrows():

        """""
         The auxiliary row is transformed into a 1xn dataframe which holds the names of the features as columns and the values in 1 row
         default behaviour for a row is to be a Series
        """""

        row_ = row.to_frame().T

        # Special case for the first event
        if first == 1:
            first = 0
            end_current_trade = row["end"]
            df1 = pd.concat([df1, row_])

        if row["start"] > end_current_trade:
            end_current_trade = row["end"]
            df1 = pd.concat([df1, row_])

    return df1



# This makes a series stationary
def fracDiff_FFD(series, d, thres=1e-5):
    # Constant width window (new solution)
    # Note 1: thres determines the cut-off weight for the window
    # Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]): continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# calculate the minimum d / float differentiator to make a series stationary

# Find min d
def plotMinFFD(df):
    from statsmodels.tsa.stattools import adfuller
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df[['close']]).resample('1D').last()  # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=1e-4 / 2)
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[0, 1]
        df2 = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]  # with critical value

    # out[['adfStat','corr']].plot(secondary_y='adfStat')
    print(out)
    return out[out['pVal'] < 0.05].index[0]


def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff() / gRaw.shift(1)

    for i in diff.index[1:]:
        if i in h.index:
            sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])

            if (sNeg < -h.loc[i]).item():
                sNeg = 0;
                tEvents.append(i)
            elif (sPos > h.loc[i]).item():
                sPos = 0;
                tEvents.append(i)
    return pd.Index(tEvents)


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None, eth=False, raw_data = pd.DataFrame()):
    # 1) get target

    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet

    trgt = trgt.squeeze()

    # 2) get t1 (max holding period)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[1]]

    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    events = events.dropna()

    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close,
                      events=events, ptSl=ptSl_, eth=eth, raw_data = raw_data)

    # events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan

    if side is None: events = events.drop('side', axis=1)
    return df0


def applyPtSlOnT1(close, events, ptSl, molecule, eth=False, raw_data = pd.DataFrame()):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs


    for loc, t1 in events_['t1'].items():  # fillna(close.index[-1]).items():


        df0 = close.loc[int(loc):int(t1),]  # path prices !! THIS WAS CORRECTED

        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns

        # out.loc[loc, 'price_start'] = df0.loc[int(loc)]
        # out.loc[loc, 'price_end'] = df0.loc[int(t1)]

        starting_row = close.index.get_loc(int(loc))

        # calculates the return of the 5 candles before the event was confirmed, will try to use it as a feature
        # df2 = close.iloc[max(starting_row - 6, 0): starting_row]
        df2 = close.loc[max(loc - 9, 0): loc]
        df2 = (df2 / close.loc[max(loc - 9, 2)] - 1)


        return_last_n = df2.iloc[-1]

        # Both SL and TP hit but TP was first
        if ((df0 > pt[loc]).any() and (df0 < sl[loc]).any() and (df0 < pt[loc]).idxmax() < (df0 > sl[loc]).idxmax()):

            index_pt_hit = (df0 > pt[loc]).idxmax()
            #print(f"The indes of the first hit PT is {index_pt_hit}, the range is {loc} - {t1}, it took {index_pt_hit - loc} trades")

            #out.loc[loc, "ret"] = df0.loc[index_pt_hit]

            # Checks if the option traded is on ETH because there the spreads are usually 0.001 if above 0.005

            if eth == False:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_pt_hit] > 0.01 else 0.0005 / close.loc[
                    loc]
            else:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_pt_hit] > 0.005 else 0.0005 / \
                                                                                                      close.loc[loc]
            out.loc[loc, "USD PNL"] = (raw_data.loc[index_pt_hit, "price"] - raw_data.loc[loc, "price"]) * raw_data.loc[index_pt_hit, "index_price"]  - min(0.125 * raw_data.loc[index_pt_hit, "price"] * raw_data.loc[index_pt_hit, "index_price"], 0.0003 * raw_data.loc[index_pt_hit, "index_price"]) - min(0.125 * raw_data.loc[loc, "price"] * raw_data.loc[loc, "index_price"], 0.0003 * raw_data.loc[loc, "index_price"])


            # Very important deduct fees from the reutrn :))))
            out.loc[loc, "ret"] = df0.loc[index_pt_hit] - out.loc[loc, 'fees']

            # print(f'Return is {round(out.loc[loc, "ret"],4)}, fees are {round(out.loc[loc, "fees"],4)}')
            # print(f"Price is {close.loc[index_pt_hit]}, after fees {close.loc[index_pt_hit] - 0.001 if close.loc[index_pt_hit] > 0.01 else 0.0005}")

            out.loc[loc, 'bin'] = 1
            #out.loc[loc, 'bin'] = 1 if (out.loc[loc, "ret"]) > 0 else 0
            out.loc[loc, 'end'] = index_pt_hit


        # Both TP and SL hit, SL was first
        elif ((df0 > pt[loc]).any() and (df0 < sl[loc]).any() and (df0 < sl[loc]).idxmax() < (df0 > pt[loc]).idxmax()):

            index_sl_hit = (df0 < sl[loc]).idxmax()

            #print(f"The indes of the first hit SL is {index_sl_hit}, the range is {loc} - {t1}, it took {index_sl_hit - loc} trades")

            if eth == False:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_sl_hit] > 0.01 else 0.0005 / close.loc[
                    loc]
            else:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_sl_hit] > 0.005 else 0.0005 / \
                                                                                                      close.loc[loc]

            out.loc[loc, "USD PNL"] = ( raw_data.loc[index_sl_hit, "price"] - raw_data.loc[loc, "price"]) * raw_data.loc[index_sl_hit, "index_price"]  - min(0.125 * raw_data.loc[index_sl_hit, "price"] * raw_data.loc[index_sl_hit, "index_price"], 0.0003 * raw_data.loc[index_sl_hit, "index_price"]) - min(0.125 * raw_data.loc[loc, "price"] * raw_data.loc[loc, "index_price"], 0.0003 * raw_data.loc[loc, "index_price"])


            out.loc[loc, "ret"] = df0.loc[index_sl_hit] + out.loc[loc, 'fees']

            # print(f'Return is {round(out.loc[loc, "ret"],4)}, fees are {round(out.loc[loc, "fees"],4)}')

            out.loc[loc, 'bin'] = -1
            #out.loc[loc, 'bin'] = -1 if (out.loc[loc, 'ret']) < 0 else 0
            out.loc[loc, 'end'] = index_sl_hit


        # Only TP was hit
        elif ((df0 > pt[loc]).any()):

            index_pt_hit = (df0 > pt[loc]).idxmax()
            #print(f"The indes of the first hit PT is {index_pt_hit}, the range is {loc} - {t1}, it took {index_pt_hit - loc} trades")

            if eth == False:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_pt_hit] > 0.01 else 0.0005 / close.loc[
                    loc]
            else:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_pt_hit] > 0.005 else 0.0005 / \
                                                                                                      close.loc[loc]

            out.loc[loc, "ret"] = df0.loc[index_pt_hit] - out.loc[loc, 'fees']


            out.loc[loc, "USD PNL"] = (raw_data.loc[index_pt_hit, "price"] - raw_data.loc[loc, "price"]) * raw_data.loc[index_pt_hit, "index_price"]  - min(0.125 * raw_data.loc[index_pt_hit, "price"] * raw_data.loc[index_pt_hit, "index_price"], 0.0003 * raw_data.loc[index_pt_hit, "index_price"]) - min(0.125 * raw_data.loc[loc, "price"] * raw_data.loc[loc, "index_price"], 0.0003 * raw_data.loc[loc, "index_price"])


            # print(f'Return is {round(out.loc[loc, "ret"],4)}, fees are {round(out.loc[loc, "fees"],4)}')

            out.loc[loc, 'bin'] = 1
            #out.loc[loc, 'bin'] = 1 if (out.loc[loc, "ret"]) > 0 else 0
            out.loc[loc, 'end'] = index_pt_hit


        # Only SL hit
        elif (df0 < sl[loc]).any():
            index_sl_hit = (df0 < sl[loc]).idxmax()

            #print(f"The indes of the first hit SL is {index_sl_hit}, the range is {loc} - {t1}, it took {index_sl_hit - loc} trades")

            if eth == False:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_sl_hit] > 0.01 else 0.0005 / close.loc[
                    loc]
            else:
                out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[index_sl_hit] > 0.005 else 0.0005 / \
                                                                                                      close.loc[loc]

            out.loc[loc, "ret"] = df0.loc[index_sl_hit] + out.loc[loc, 'fees']

            out.loc[loc, "USD PNL"] = (raw_data.loc[index_sl_hit, "price"] - raw_data.loc[loc, "price"]) * raw_data.loc[index_sl_hit, "index_price"]  - min(0.125 * raw_data.loc[index_sl_hit, "price"] * raw_data.loc[index_sl_hit, "index_price"], 0.0003 * raw_data.loc[index_sl_hit, "index_price"]) - min(0.125 * raw_data.loc[loc, "price"] * raw_data.loc[loc, "index_price"], 0.0003 * raw_data.loc[loc, "index_price"])


            # print(f'Return is {round(out.loc[loc, "ret"],4)}, fees are {round(out.loc[loc, "fees"],4)}')

            #out.loc[loc, 'bin'] = -1 if (out.loc[loc, "ret"]) < 0 else 0
            out.loc[loc, "bin"] = -1

            out.loc[loc, 'end'] = index_sl_hit


        else:
            try:
                event_return = (close.loc[t1] - close.loc[loc]) / close.loc[loc]

                if eth == False:
                    out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[t1] > 0.01 else 0.0005 / \
                                                                                                         close.loc[
                                                                                                             loc]
                else:
                    out.loc[loc, 'fees'] = 0.001 / close.loc[loc] if close.loc[t1] > 0.005 else 0.0005 / \
                                                                                                          close.loc[loc]



                out.loc[loc, "USD PNL"] = (raw_data.loc[t1, "price"] - raw_data.loc[loc, "price"]) * \
                                          raw_data.loc[t1, "index_price"] - min(
                    0.125 * raw_data.loc[t1, "price"] * raw_data.loc[t1, "index_price"],
                    0.0003 * raw_data.loc[t1, "index_price"]) - min(
                    0.125 * raw_data.loc[loc, "price"] * raw_data.loc[loc, "index_price"],
                    0.0003 * raw_data.loc[loc, "index_price"])


                out.loc[loc, "ret"] = event_return
                out.loc[loc, "bin"] = 1 if event_return >= 0 else -1
                out.loc[loc, "end"] = t1
            except:
                print(df0)
                print(loc)
                print(t1)
                print(close.loc[loc])
                print(close.loc[t1])
                sys.exit()

        out.loc[loc, 'start'] = loc
        out.loc[loc, 'volatility'] = events_.loc[int(loc), 'trgt']

        out.loc[loc, 'ret_last_10'] = return_last_n

    # Since there was an issue where the random forest produced only 1.0 predictions, I found out that
    # OF COURSE IT WOULD NOT all the returns for bin = -1 are negative and not multiplied by their direction
    #out["ret_train"] = out["ret"].apply(lambda x: abs(x))
    return out


# If we used float differentiation to fit the Random Forst + Bagging Classifier, we need to
# see how it performs on the raw data as it is there that it enters positions, i.e. generate a signal on
# the float differentiated data and see how it performs on the actual market data
def results_from_raw_data(events_df, raw_df, ptSl=[2, 1]):
    raw_df['volatility'] = getDailyVol(raw_df['close'])

    events_df.set_index('t1', inplace=True)

    out = events_df[['start']].copy(deep=True)

    for t1, row in events_df.iterrows():

        close_series = raw_df[row['start']:t1]
        close_returns = close_series['close'] / close_series.loc[row['start'], 'close']

        side = events_df.loc[t1, 'prediction']
        takeprofit = ptSl[0] * raw_df.loc[row['start'], 'volatility']
        stoploss = - ptSl[1] * raw_df.loc[row['start'], 'volatility']

        if ((side * close_returns) < stoploss).any():
            out.loc[row['start'], 'return'] = stoploss
            out.loc[row['start'], 'bool'] = False

        elif ((side * close_returns) > takeprofit).any():
            out.loc[row['start'], 'return'] = takeprofit
            out.loc[row['start'], 'bool'] = True

        else:
            out.loc[row['start'], 'return'] = side * close_returns.iloc[-1]
            out.loc[row['start'], 'bool'] = True if out.loc[row['start'], 'return'] > 0 else False

    out['return'] = 1 + out['return']
    out['result'] = out['return'].cumprod()

    print(f"Accuracy given the raw dataset, and training on float differentiated data is {out['bool'].mean()}")
    plt.plot(out['result'])
    plt.show()


# Get red of overlaps between train and test set when doing cross validation

def getTrainTimes(t1, testTimes):
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index  # train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index  # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getIndMatrix(barIx, t1):
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.items()): indM.loc[t0:t1, i] = 1.
    return indM


def getAvgUniqueness(indM):
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU


# This is basically what is refered to as principle-component analysis
# Reduces the dimensionality of X by dropping features with small eigen values
# Tries to keep the variance while reducing the number of variables

def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def orthoFeats(dfX, varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


def continiousOrthoFeats(df, varThres=.95):
    results = []

    for i in range(1, len(df) + 1):
        if i < len(df.columns) + len(df.columns) * len(df.columns):
            results.append(df.iloc[-1])

        else:
            subset = df.iloc[:i]
            transformed = orthoFeats(subset, varThres)
            transformed = pd.DataFrame(transformed, index=subset.index, columns=df.columns)

            results.append(transformed.iloc[-1])

    return pd.DataFrame(results, index=df.index, columns=df.columns)


from sklearn.model_selection._split import _BaseKFold


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def binHR(sl, pt, freq, tSR):
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    p = (- b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)
    return p


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
    # Random draws from a mixture of gaussians
    ret1 = np.random.normal(mu1, sigma1, size=int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size=int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret


# Probability that a stategy will fail
def probFailure(ret, freq, tSR):
    # Derive probability that strategy may fail
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = binHR(rNeg, rPos, freq, tSR)
    risk = ss.norm.cdf(thresP, p, p * (1 - p))  # approximation to bootstrap
    return risk


# An estimate for daily volatility, using exponential smoothening
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


# Multiprocessing function for pandas

# Auxiliary functions
def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None: task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close();
    pool.join()  # this is needed to prevent memory leaks
    return out


def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


# This is what allows multiprocessing
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''

    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out: df0 = pd.concat(out,
                                  ignore_index=True)  # for i in out: df0 = pd.concat([df0, i], ignore_index= True)#for i in out:df0=df0.append(i)
    df0 = df0.sort_index()
    return df0


def monte_carlo_permutation_generator(df):
    columns_to_log = ['open', 'high', 'low', 'close']
    df[columns_to_log] = df[columns_to_log].apply(np.log)
    df.loc[:, 'close-open'] = df['close'] - df['open']
    df.loc[:, 'high-open'] = df['high'] - df['open']
    df.loc[:, 'low-open'] = df['low'] - df['open']
    df.loc[:, 'open-close(-1)'] = df['open'] - df['close'].shift(-1)

    columns_to_shuffle = ['close-open', 'high-open', 'low-open', 'open-close(-1)']
    df.loc[:, columns_to_shuffle] = df[columns_to_shuffle].apply(np.random.permutation)

    first_open = df['open'].iloc[0]
    df.loc[:, 'open'] = df['open'].shift(1) + df['open-close(-1)']
    df.loc[df.index[0], 'open'] = first_open
    df.loc[:, 'high'] = df['open'] + df['high-open']
    df.loc[:, 'close'] = df['open'] + df['close-open']
    df.loc[:, 'low'] = df['open'] + df['low-open']

    df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(np.exp).round(2)
    return df


if __name__ == '__main__':
    # Normal Test
    df = pd.read_csv("btc_27_09_2024_call_70000.csv", parse_dates=True)

    if "contracts" not in df.columns:
        df["contracts"] = df["amount"]

    df = df[['timestamp', 'tick_direction', 'price', 'mark_price', 'iv', 'index_price', 'direction', 'contracts', "block_trade_id", "instrument_name","combo_trade_id"]]#, "liquidation"]]

    main(df, eth=False)

    # Monte Cardo Test
    df = monte_carlo_permutation_generator(df)
    main(df)