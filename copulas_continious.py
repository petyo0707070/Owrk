import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from copulae.elliptical import StudentCopula
from copulae.archimedean import GumbelCopula, ClaytonCopula, FrankCopula
import plotly.graph_objs as go
import seaborn as sns
pd.options.mode.chained_assignment = None

def main():

    # Load the data
    iygj = pd.read_csv("iygj.csv", header = 0, index_col = 0)
    qw9a = pd.read_csv("qw9a.csv", header = 0, index_col=0)


    # Drop the NA Values which are a lot unfortunately
    iygj = iygj.dropna()
    qw9a = qw9a.dropna()



    # Drop the dates that are present only in one of the two dataframes
    qw9a = qw9a[qw9a["date"].isin(iygj["date"])]


    # Reset the index
    iygj = iygj.reset_index(drop = True)
    qw9a = qw9a.reset_index(drop = True)

    # Calculate the returns and set the index as a date
    iygj["date"] = pd.to_datetime(iygj["date"])
    iygj.set_index("date", inplace= True)

    qw9a["date"] = pd.to_datetime(qw9a["date"])
    qw9a.set_index("date", inplace= True)

    iygj = iygj.dropna()
    qw9a = qw9a.dropna()


    #Split the data 64% for training, 18 % for validation and 18% for testing
    iygj_validation_test = iygj[int(0.64 * len(iygj)):]
    iygj_validation = iygj_validation_test[:int(0.5 * len(iygj_validation_test))]
    iygj_test = iygj_validation_test[int(0.5 * len(iygj_validation_test)):]
    del iygj_validation_test
    iygj = iygj[:int(0.64 * len(iygj))]

    qw9a_validation_test = qw9a[int(0.64 * len(qw9a)):]
    qw9a_validation = qw9a_validation_test[:int(0.5 * len(qw9a_validation_test))]
    qw9a_test = qw9a_validation_test[int(0.5 * len(qw9a_validation_test)):]
    del qw9a_validation_test
    qw9a = qw9a[:int(0.64 * len(qw9a))]



    df = pd.merge(iygj["SP039"], qw9a["SP039"], left_index=True, right_index=True)

    df['x'] = iygj["SP039"]
    df['y'] = qw9a['SP039']
    df['spread'] = df['x'] - df['y']

    #run_training(df)



    # Join the 2 dfs in a single dataframe
    df_validation = pd.merge(iygj_validation["SP039"], qw9a_validation["SP039"], left_index=True, right_index=True)
    print(df_validation)
    df_validation['x'] = iygj_validation["SP039"]
    df_validation['y'] = qw9a_validation['SP039']
    df_validation['spread'] = df_validation['x'] - df_validation['y']
    #run_training(df_validation)


    # Run on the test dataset
    df_test = pd.merge(iygj_test["SP039"], qw9a_test["SP039"], left_index=True, right_index=True)
    print(df_test)
    df_test['x'] = iygj_test["SP039"]
    df_test['y'] = qw9a_test['SP039']
    df_test['spread'] = df_test['x'] - df_test['y']
    run_training(df_test)

    sys.exit()




    plt.scatter(iygj["U"], qw9a["U"])
    plt.title("Scatterplot cdfs")
    plt.show()


    sns.jointplot(data = df, x = "U_x", y = "U_y", kind = "kde", height=6)
    plt.show()

    sys.exit()

# The timeseries are NOT cointegrated
def run_cointegration_test(y, x):
    import statsmodels.api as sm
    import statsmodels.tsa.stattools as ts


    x_y_df = pd.merge(x,y, left_index = True, right_index= True)

    x1 = x_y_df["PX_LAST_x"]
    y1 = x_y_df["PX_LAST_y"]

    result = ts.coint(y1, x1, trend = "ctt", maxlag= 5, autolag="t-stat")
    print(result)

def plot_probability_distributions(x, y):
    from scipy.stats import gaussian_kde
    from numpy import linspace

    kernel_x = gaussian_kde(x.values)
    dist_space_x = linspace(min(x.values), max(x.values), 100)

    kernel_y = gaussian_kde(y.values)
    dist_space_y = linspace(min(y.values), max(x.values), 100)

    plt.plot(dist_space_x, kernel_x(dist_space_x), label = "IYGJ (Covered)")
    plt.plot(dist_space_y, kernel_y(dist_space_y), label = "QW9A (SSA)")
    plt.legend()

    plt.show()


# inverse cdf i.e. quantile function
def quantile(x, q):
    try:
        sample = x.values
    except:
        sample = x
    sample = sorted(sample)
    return sample[int(len(sample) * q)]

# Evaluating correlation dependence is needed when looking which copula to select
# The idea is that we reduce q and estimate coefficients of correlation with q going towards 0
# and we check how the correlation evolves. Important this needs to be done for the lower tail for both combined
# and separetely for the upper tail for both combined

"""
First important finding is that the tails to the downside are almost perfectly correlated,
while to the upside it is volatile , lowest it goes is 90% but still good  to know
"""
def evaluate_correlation_dependence(x, y):
    # Input needs to be a dataframe unfortunately

    lower_tail_correlations = []
    upper_tail_corrleations = []

    #This will be used for the labels
    x_label_positions = [i for i in range(1,50,5)]
    x_labels = [i/1000 for i in range(1,50,5)]

    for q in range(1, 50):
        q = q / 1000
        lower_tail_correlations.append(x["return"][0:int(q * len(x) )].corr
                                       (y["return"][0:int(q * len(y) )]))
        upper_tail_corrleations.append(x["return"][int( (1 - q) * len(x) ):].corr
                                       (y["return"][int( (1 -q)  * len(y) ):]))


    plt.plot(lower_tail_correlations, label = "Lower tail")
    plt.plot(upper_tail_corrleations, label = "Upper tail")
    #plt.plot(2 - pow(2, 1/q), label = "Theoretical Gumbel Dependence")

    plt.title("Tail Correlations based on q")
    plt.xticks(x_label_positions, x_labels)
    plt.legend()
    plt.show()

# This will be the more accurate way to select a copula, correlations are not sufficient
def evaluate_tail_dependence(x, y):
    pass


def choose_best_copula(df):

    highest_likelihood = {"ML": 0, "Copula": None}

    for copula in [GumbelCopula, ClaytonCopula, FrankCopula, StudentCopula]:

        if copula == StudentCopula:
            temp_copula = copula()
            temp_copula.fit(data = df, to_pobs= False, verbose = 0)
        else:
            temp_copula = copula()
            temp_copula.fit(data = df, to_pobs= False, verbose = 0)

        if temp_copula.log_lik(df.to_numpy(), to_pobs=False) > highest_likelihood["ML"]:
            highest_likelihood["ML"] = temp_copula.log_lik(df.to_numpy())
            highest_likelihood["Copula"] = temp_copula

    return highest_likelihood


def run_training(df):


    #df["Conditional Probability"] = copula.cdf(df[["U_x", "U_y"]])

    df = df[["x", "y", "spread"]]

    df["Date"] = df.index.copy()

    # A variable that is -1 if we are short the spread (X - Y, when the conditional probability is <5%)
    # the position is 1 if we are long the spread (X - Y, wihhc is when the conditional probability is 95%)
    position = 0

    # This tracks when the position was entered
    index_entered = 0

    # An array that will hold whether the trades were accurate or not
    outcomes = []

    # This will hold the years that have passed intended to be used for the retraining
    years_passed = []

    # Initiate a variable that will hold the copula, needed as we are operating in a continiously updated for loop
    copula = None
    conditional_probability = None

    for index, row in df.iterrows():


        # Only append the year when there is no previous year
        if row["Date"].year not in years_passed and len(years_passed) < 1:
            years_passed.append(row["Date"].year)

        """"
        This will be initiated if this is the fist day of the year
        i.e. needs retraining since we are doing a rolling window
        """""
        if row["Date"].year not in years_passed and len(years_passed) >= 1:

            pos = df.index.get_loc(index)
            years_passed.append(row["Date"].year)

            # This is initiated as vanilla i.e. take the last 252 days /1 trading year/
            if pos >= 252:
                temp_df = df.iloc[pos - 252: pos]

            # This will only come into play if it is an year which is in the begining and we do not yet have 252 days
            else:
                temp_df = df.iloc[0 : pos]

            print(temp_df)

            # Get the Empirical CDF of the last year
            cdf_iygj = ECDF(temp_df["x"].values)
            cdf_qw9a = ECDF(temp_df["y"].values)

            # Get their cdf
            temp_df["U_x"] = temp_df["x"].apply(lambda x: cdf_iygj(x))
            temp_df["U_y"] = temp_df["y"].apply(lambda x: cdf_qw9a(x))

            # Select the most appropriate copula
            result = choose_best_copula(temp_df[["U_x", "U_y"]])
            copula = result["Copula"]


            row["U_x"] = cdf_iygj(row["x"])
            row["U_y"] = cdf_qw9a(row["y"])

            # Calculate the conditional probability, has to be passed as a dataframe not a series
            conditional_probability = copula.cdf(pd.DataFrame({
                                        "U_x": [row["U_x"]],
                                        "U_y": [row["U_y"]]}))


        # The default operation when doing the walk forward
        if row["Date"].year in years_passed and len(years_passed) >= 2:

            row["U_x"] = cdf_iygj(row["x"])
            row["U_y"] = cdf_qw9a(row["y"])
            conditional_probability = copula.cdf(pd.DataFrame({
                                        "U_x": [row["U_x"]],
                                        "U_y": [row["U_y"]]}))


            # Short the spread
            if position == 0 and conditional_probability < 0.025:
                position = 1
                index_entered = index

            # Long the spread
            if position == 0 and conditional_probability > 0.975:
                position = -1
                index_entered = index

            print(f"Right now position is {position} the conditional probability is {conditional_probability}, spread is {row['spread']} ")

            # Close the short spread
            if position == -1 and conditional_probability <=  0.5:
                position = 0
                outcomes.append(1 if df.loc[index_entered, "spread"] > row["spread"] else 0)
                index_entered = 0



            # Close the long spread
            if position == 1 and conditional_probability >= 0.50:
                position = 0
                outcomes.append(1 if df.loc[index_entered, "spread"] < row["spread"] else 0)
                index_entered = 0



    print(outcomes)
    print(sum(outcomes) / len(outcomes))




if __name__ == "__main__":
    main()