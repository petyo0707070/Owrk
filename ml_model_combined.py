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

def main():
    X_train = pd.read_csv("x_train.csv")
    X_validation = pd.read_csv("x_validation.csv")
    X_test = pd.read_csv("x_test.csv")

    y_train = pd.read_csv("y_train.csv")
    y_validation = pd.read_csv("y_validation.csv")
    y_test = pd.read_csv("y_test.csv")




    model = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False,
                                           class_weight='balanced_subsample',
                                           min_weight_fraction_leaf=0.05, n_jobs = 1)
    model = BaggingClassifier(estimator=model, n_estimators = 100)


    #walk_forward_result = evaluate_model(model, X_train, y_train.values.ravel())
    #print(f"The result of the walk forward are {walk_forward_result}")



    model.fit(X_train, y_train.values.ravel())

    save_model(model)

    run_testing(X_train, y_train, model, "train")
    run_testing(X_validation, y_validation, model,"validation")
    run_testing(X_test, y_test, model, "test")



def save_model(model):
    import joblib
    joblib.dump(model, "tyranid_combined.pkl")


def run_testing(X, y, model, type: str):
    y_pred_train = model.predict(X)

    y = y.to_numpy().flatten()


    result_training = y_pred_train == y
    precision_train, recall_train, fscore_train, support_train= score(y, y_pred_train)
    print(f"Precision for the {type} set is {precision_train}")

    """""
    df = pd.DataFrame()
    df["bool"] = result_training
    df["prediction"] = y_pred_train
    df["result"] = np.where(df['prediction'] == 1, (df['ret'] * 1 + 1),
                                 np.where(df['prediction'] == - 1, ((df['ret'] * - 1) + 1), 1))
    df['result'] = df['result'].cumprod()
    df['result'].plot(kind='line', title=f'Returns of the RF Bagged Model on the {type} set')
    plt.show()
    """""

def evaluate_model(model, X, y):

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model.fit(X, y)


    cv = StratifiedKFold(n_splits = 5)
    score = cross_val_score(model, X, y, scoring = "precision_weighted", cv = cv, n_jobs = 16, error_score = "raise")

    return score

if __name__ == "__main__":
    main()