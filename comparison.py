import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support as score


df = pd.read_csv("comparison.csv")
model = joblib.load("tyranid_combined.pkl")

X = df[["volatility", "ret_last_10", "iv_last_trade", "price_mark_price_deviation", "volume_last_10_trades", "buy_last_10_trades", "index_return", "abnormal_trade_conditions", "time_elapsed" ]]
print(X)
y = df["bin"]
y_pred = model.predict(X)


precision, recall, fscore, support = score(y, y_pred)

print(y_pred)