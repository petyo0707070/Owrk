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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as ss


def main(df):

    print(df)

    differentiator = plotMinFFD(df)
    stationary_series = fracDiff_FFD((df[['close']]), differentiator, 1e-4 / 2)
    #stationary_series = df


    sigma = getDailyVol(stationary_series['close'], 100)

    stationary_series = stationary_series[stationary_series.index.isin(sigma.index)]
    events = getTEvents(stationary_series['close'], 2 * sigma)

    t1 = stationary_series['close'].index.searchsorted(events + pd.Timedelta(days=5))
    t1 = t1[t1 < stationary_series['close'].shape[0]]
    t1 = pd.Series(stationary_series['close'].index[t1], index=events[:t1.shape[0]])

    df1 = getEvents(stationary_series['close'], events, [2, 1], 2 * sigma, 0.004, 48, t1, None)

    # This one is useful if you want to add a feature and you calculate it on the initial stationary series this maps the feature onto the dataset witj the events
    #df1["hawkes"] = df1["t1"].apply(lambda x: stationary_series.loc[x, "hawkes"])
    print(df1)

    # This implements PCA / Principal Component Analysis/ even before we train the model in an attempt to make it less overfit

    X_train, X_test, y_train, y_test = train_test_split(df1[['volatility', 'ret last 5', 'bullish last 5']], df1['bin'], test_size=0.3, shuffle=False)


    orthogonal_features_train = orthoFeats(X_train)
    #X_train = pd.DataFrame(orthogonal_features_train, columns=['volatility', 'ret last 5', 'bullish last 5'])

    #orthogonal_features_test = continiousOrthoFeats(df1[['volatility', 'ret last 5', 'bullish last 5']])
    #X_test = pd.DataFrame(orthogonal_features_test, columns=['volatility', 'ret last 5', 'bullish last 5']).loc[X_test.index]

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
    def random_forest(i=0, avgU=avgU):

        if i == 0:
            model = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', criterion='entropy')

        elif i == 1:
            model = DecisionTreeClassifier(criterion='entropy', max_features=0.75, class_weight='balanced')
            model = BaggingClassifier(estimator=model, n_estimators=1000, max_samples=avgU)

        elif i == 2:
            model = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                           class_weight='balanced_subsample',
                                           min_weight_fraction_leaf=0.05)  # , max_features= int(1))
            model = BaggingClassifier(estimator=model, n_estimators=1000, max_samples=avgU)  # , max_features=1.)

        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=False)
        cross_score = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring="f1")
        print(f"Cross-validation score for the random-forest model is {np.mean(cross_score)}")




        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        simple_stats(y_pred)

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

        # grid = grid_search(RandomForestClassifier(criterion= 'entropy', bootstrap=False, class_weight='balanced_subsample', min_weight_fraction_leaf=0.05, max_features= int(1)))
        # grid.fit(X_train, y_train)
        # params = grid.best_params_
        # print(params)

        # model_1 = RandomForestClassifier( criterion= 'entropy', bootstrap=False, class_weight='balanced_subsample', min_weight_fraction_leaf=0.05,
        #                                        max_features= int(1), n_estimators = 1, max_depth = params['max_depth'],
        #                                        min_samples_split= params['min_samples_split'], min_samples_leaf= params['min_samples_leaf'])

        # model_1 = BaggingClassifier(estimator=model, n_estimators= 100, max_samples=avgU, max_features=1.)
        # model_1.fit(X_train, y_train)
        # y_pred = model_1.predict(X_test)
        # print(f"Grid search results: Best parametres are {grid.best_params_} with cross-validation score of {grid.best_score_}")
        # simple_stats(model_1)

    def simple_stats(y_pred, df1=df1):
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"The accuracy of the model with normal bootstrapping is {accuracy}")
        result = y_pred == y_test
        df2 = df1.loc[y_test.index]
        df2['bool'] = result

        df2['result'] = np.where(df2['bool'], (df2['ret'].abs() + 1), (- df2['ret'].abs() + 1))
        df2['result'] = df2['result'].cumprod()
        df2['result'].plot(kind='line', title='Returns of the Bagging Classifier over time')
        plt.show()

    def grid_search(model):
        param_grid = {
            'max_depth': [None, 5, 10],  # Maximum depth of each tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='f1', n_jobs=-1)
        return grid_search

    random_forest(2)

    # indMatrix = getIndMatrix(df.index, t1)
    # standard_uniqueness = np.random.choice(indMatrix.columns, size = 50)
    # bootstrap_result = seqBootstrap(indMatrix, 50)

    # print(f"Standard uniqueness {getAvgUniqueness(indMatrix[standard_uniqueness]).mean()}")
    # print(f"Sequential Bootstrap uniqueness {getAvgUniqueness(indMatrix[bootstrap_result]).mean()}")
    # print(indMatrix[bootstrap_result])


def getWeights_FFD(d, thres):
    w, k = [1.], 1

    while True:

        w_ = -w[-1] / k * (d - k + 1)

        if abs(w_) < thres: break

        w.append(w_);
        k += 1

    return np.array(w[::-1]).reshape(-1, 1)


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

            if sNeg < -h.loc[i]:
                sNeg = 0;
                tEvents.append(i)
            elif sPos > h.loc[i]:
                sPos = 0;
                tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet

    # 2) get t1 (max holding period)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]

    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    events = events.dropna()

    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close,
                      events=events, ptSl=ptSl_)

    # events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan

    if side is None: events = events.drop('side', axis=1)
    return df0


def applyPtSlOnT1(close, events, ptSl, molecule):
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

        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns


        starting_row = close.index.get_loc(t1)

        # calculates the return of the 5 candles before the event was confirmed, will try to use it as a feature
        #df2 = close.iloc[max(starting_row - 6, 0): starting_row]
        df2 = close.iloc[starting_row - 5: starting_row ]
        df2 = (df2 / close.iloc[starting_row - 5] - 1)

        print(sl.loc[loc])
        print(pt.loc[loc])
        print(df2)
        return_last_5 = df2.iloc[-1]
        bullish_last_5 = (close.iloc[max(starting_row - 5, 0): starting_row].diff() > 0).sum()

        if (df0 < sl[loc]).any():
            out.loc[loc, "ret"] = sl[loc]
            out.loc[loc, "bin"] = -1

        elif (df0 > pt[loc]).any():
            out.loc[loc, "ret"] = pt[loc]
            out.loc[loc, 'bin'] = 1

        else:
            out.loc[loc, "ret"] = df0.iloc[-1]
            out.loc[loc, "bin"] = np.sign(out.loc[loc, "ret"])

        out.loc[loc, 'volatility'] = events_.loc[loc, 'trgt']
        out.loc[loc, 'ret last 5'] = return_last_5
        out.loc[loc, 'bullish last 5'] = bullish_last_5

    return out


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


def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None: sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        if len(phi) % 10 == 0: print(len(phi))
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]

        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


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


# Grid search with Purged K-Fold Cross-Validation
def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], n_jobs=-1, pctEmbargo=0,
                **fit_params):
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'  # f1 for meta-labeling

    else:
        scoring = 'neg_log_loss'  # symmetric towards all cases

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged

    gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params \
            [gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


# Derive position size base on probability usin w = 2 * Z - 1
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    if prob.shape[0] == 0: return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** .5  # t-value of OvR
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal=side*size

    if 'side' in events: signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


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
    df = pd.read_csv(r"C:\Users\A96RICB\pythonProject\trading_view_eur_usd_daily.csv",  names = ["time", "open", "high", "low", "close", "volume"],index_col=0, parse_dates=True)

    #df = pd.read_csv('M:\OE0855\PB\Repository\R_code\itrax_prices.csv', usecols=[1,2],  index_col=0, parse_dates=True)
    #df.rename(columns = {"PX_LAST": "close"}, inplace= True)

    main(df)

    # Monte Cardo Test
    df = monte_carlo_permutation_generator(df)
    main(df)
