import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from copulae.elliptical import StudentCopula
from copulae.archimedean import GumbelCopula, ClaytonCopula, FrankCopula
import plotly.graph_objs as go
import seaborn as sns


def main():

    # Load the data
    iygj = pd.read_csv("iygj.csv", header = 0, index_col = 0)
    qw9a = pd.read_csv("qw9a.csv", header = 0, index_col=0)


    # Drop the NA Values which are a lot unfortunately
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


    #plot_probability_distributions(iygj["SP039"], qw9a["SP039"])

    # This fits a cdf for the empirical distribution of the training sets for the 2 indexes
    cdf_iygj = ECDF(iygj["SP039"].values)
    cdf_qw9a = ECDF(qw9a["SP039"].values)

    # The idea of the copula strategy is to go long A and short B
    # when cdf(A) = 5% and cdf(B) = 95% and vice versa, percentages are subject to the
    # confidence level we choose

    iygj["U"] = iygj["SP039"].apply(lambda x: cdf_iygj(x))
    qw9a["U"] = qw9a["SP039"].apply(lambda x: cdf_qw9a(x))

    df = pd.merge(iygj["U"], qw9a["U"], left_index=True, right_index=True)


    """
    cdf_iygb_prices = ECDF(iygb["PX_LAST"].values)
    cdf_qw9a_prices = ECDF(qw9a["PX_LAST"].values)
    iygb["U_1"] = iygb["PX_LAST"].apply(lambda x: cdf_iygb_prices(x))
    qw9a["U_1"] = qw9a["PX_LAST"].apply(lambda x: cdf_qw9a_prices(x))
    plt.scatter(iygb["U_1"], qw9a["U_1"])
    plt.title("Scatterplot prices")
    plt.show()
    sys.exit()
    """

    result = choose_best_copula(df)
    copula = result["Copula"]


    print(copula)
    print(copula.params)

    df['x'] = iygj["SP039"]
    df['y'] = qw9a['SP039']
    df['spread'] = df['x'] - df['y']

    run_training(df, copula)



    # This codes will run the validation
        # First convert the raw values to their quantiles
    iygj_validation["U"] = iygj_validation["SP039"].apply(lambda x: cdf_iygj(x))
    qw9a_validation["U"] = qw9a_validation["SP039"].apply(lambda x: cdf_qw9a(x))

    # Join the 2 dfs in a single dataframe
    df_validation = pd.merge(iygj_validation["U"], qw9a_validation["U"], left_index=True, right_index=True)
    df_validation['x'] = iygj_validation["SP039"]
    df_validation['y'] = qw9a_validation['SP039']
    df_validation['spread'] = df_validation['x'] - df_validation['y']
    run_training(df_validation, copula)

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
            temp_copula.fit(data = df, to_pobs= False)
        else:
            temp_copula = copula()
            temp_copula.fit(data = df, to_pobs= False)

        if temp_copula.log_lik(df.to_numpy(), to_pobs=False) > highest_likelihood["ML"]:
            highest_likelihood["ML"] = temp_copula.log_lik(df.to_numpy())
            highest_likelihood["Copula"] = temp_copula

    return highest_likelihood


def run_training(df, copula):

    df["Conditional Probability"] = copula.cdf(df[["U_x", "U_y"]])

    # A variable that is -1 if we are short the spread (X - Y, when the conditional probability is <5%)
    # the position is 1 if we are long the spread (X - Y, wihhc is when the conditional probability is 95%)
    position = 0

    # This tracks when the position was entered
    index_entered = 0

    # An array that will hold whether the trades were accurate or not
    outcomes = []


    for index, row in df.iterrows():


        # Short the spread
        if position == 0 and row["Conditional Probability"] < 0.05:
            position = -1
            index_entered = index

        # long the spread
        if position == 0 and row["Conditional Probability"] > 0.95:
            position = 1
            index_entered = index

        print(f"Right now position is {position} the conditional probability is {row['Conditional Probability']}, spread is {row['spread']} ")

        # Close the short spread
        if position == -1 and row['Conditional Probability'] >=  0.2:
            position = 0
            outcomes.append(1 if df.loc[index_entered, "spread"] > row["spread"] else 0)
            index_entered = 0



        # Close the long spread
        if position == 1 and row['Conditional Probability'] <= 0.80:
            position = 0
            outcomes.append(1 if df.loc[index_entered, "spread"] < row["spread"] else 0)
            index_entered = 0



    print(df["Conditional Probability"].describe())
    print(outcomes)



if __name__ == "__main__":
    main()