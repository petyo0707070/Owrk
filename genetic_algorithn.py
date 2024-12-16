import pandas as pd
import numpy as np
import matplotlib as plt
import operator
import math
import random

df = pd.read_csv('trading_view_btc_1hr.csv', names = ['Time', 'Open', 'High', 'Low','Close', 'Volume'])

train_df = df[:round(0.8 * len(df))]
test_df = df[round(0.7 * len(df)): ]
#old_df = df[round(0.8 * len(df)):]
print(df)

# There are 2 Hyperparametres - pattern size i.e. how many comparisons and candle lag (how much back can we check 3 means current candle and the 2 previous ones)
pattern_size = 3    
candle_lag = 3

# The settings used
population_size = 200 # Each generation has 200 members
elitism = 1 # The best performer of each generation will automatically pass on its genes to the next population without any chance for mutation either gene or pattern one
mutation_chance = 0.05 # A random chance that changes only one gene i.e H -> L or H[0] -> H[1], not sure about H[0] > L[1] -> H[0] < L[1]
fresh_pattern_chance = 0.02 # A random chance that changes a whole and block in a gene pool
minimum_frequency = 0.025 # The pattern needs to be present in at least 2.5% of the population to be viable
max_generation_number = 10 # A variable that limites the total number of generations
total_runs = 5 # The number of distinct runs i.e. uncorrelated patters we want to find
longs = 0 # 1 if we will generate long entries, 0 if we will generate short positions

best_performer = [{5*pattern_size + 1 : 0} for i in range(total_runs)]

return_distribution_best_performer = []
all_return_distributions = []

candle_genes = {
                0 : 'Open',
                1 : 'High',
                2 : 'Low',
                3: 'Close'
}

lag_genes = {i: i for i in range(0, candle_lag)}

comparison_genes = { 0: operator.gt,
                    1: operator.lt}

# A set of functions that defines the possible genes that can make up a chromozome, I am not sure about the terminology
# but for this case I will use each fully, since we are studying candlestick patterns it can look like this:
# C[0] > H[1] & L[1] < L[2] & O[2] > H[0], in this case there will be 15 different itteration


# Create a list of dictionaries that holds all the information for current generation
def create_generation(population_size = population_size, pattern_size = pattern_size):
    # Here each dictionary key corresponds to a gene, where the first will be the unique identifier of the gene and the last its fitness
    generation_list_dict = [{i: None for i in range (5 * pattern_size + 2)} for j in range(population_size)]
    return generation_list_dict

def get_candle_charesteristic_keys(pattern_size = pattern_size):
    first = 1
    candle_charesteristic_keys = []
    for i in range(2 * pattern_size):
        if i == 0:
            candle_charesteristic_keys.append(first)
        else:
            if len(candle_charesteristic_keys) % 2 == 0:
                first += 2
                candle_charesteristic_keys.append(first)
            
            else:
                first += 3
                candle_charesteristic_keys.append(first)
    
    return candle_charesteristic_keys

def get_and_positions(pattern_size = pattern_size):
    and_positions_list = []
    for i in range(1, pattern_size):
        if i == 1:
            and_positions_list.append(6)
        else:
            and_positions_list.append(6 * i -1)
    
    return and_positions_list


def strategy_robustness_check(gd):
    robust = 1

    # Here those conditions insure that we do not get strategies where we compare the low against the high  of the same candle or stuff which is impossible,
    # that leads to logical inconsistencies which lead to 0 trades

    for i in range(pattern_size):

        # No comparison between the same charecteristic of a candle for the same lag
        if gd[5 * i + 1] == gd[5*i + 4] and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0


        # We can't compare the high and low of the same candle
        if (gd[5*i + 1] == 'High' and gd[5 * i + 4] == 'Low') or (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0
        

        # No high against open for the same candle
        if (gd[5*i + 1] == 'High' and gd[5 * i + 4] == 'Open') or (gd[5*i + 1] == 'Open' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0

        
        # No high against close for the same candle
        if (gd[5*i + 1] == 'High' and gd[5 * i + 4] == 'Close') or (gd[5*i + 1] == 'Close' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0
        

        # No low against open for the same candle
        if (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'Open') or (gd[5*i + 1] == 'Open' and gd[5 * i + 4] == 'Low') and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0


        # No low against close for the same candle
        if (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'Close') or (gd[5*i + 1] == 'Close' and gd[5 * i + 4] == 'Low') and gd[5*i + 2] == gd[5*i + 5]:
            robust = 0    

        # No open against previous candle high, first variaton
        if (gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == -1):
            robust = 0

        
        # No open against previous candle high, second variation
        if (gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1):
            robust = 0


        # No open against previous candle low, first variation
        if gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == -1:
            robust = 0


        # No open against previous candle low, second variation
        if gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1:
            robust = 0


        # No open against previous candle close, first variation
        if gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'Close' and gd[5*i + 2] - gd[5*i + 5] == -1:
            robust = 0


        # No open against previous candle close, second varion
        if gd[5*i + 1] == 'Close' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1:
            robust = 0


        # No high of current gainst low of previous candle, first variation
        if gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == -1:
            robust = 0


        # NO high of cuurent candle against low of previous candle, second variation        
        if gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == 1:
            robust = 0  

        # No low of current candle against high of previous candle, first variation   
        if gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == -1:
            robust = 0

        # No low of current candle against high of previous candle, second variation   
        if gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == 1:
            robust = 0

        #print(robust)
        if robust == 0:
            while robust == 0:
                # The idea is that if the gene of this strategy does not satisfy the above-mentioned conditions we need to go on
                # generating new variations until we get a robust one, to do that we starta new random assignment procedure for the current
                # condition i.e. C[2] > H[2] is regenerated and reitterated over all conditions
                candle_charecteristic_keys = [5 * i + 1, 5 * i + 4]
                lag_keys = [5 * i + 2, 5 * i + 5]
                candle_charecteristic_values = np.random.choice(list(candle_genes.values()), size=len(candle_charecteristic_keys), replace=True)
                lag_values = np.random.choice(list(lag_genes.values()), size = len(lag_keys), replace = True)

                # New conditions have been created now they have to be rechecked :)
                # It is a very long if-chain where the principle is guilty until proven otherwise, i.e. all conditions need to be cycled through
                #Until we can confidently declare that robust = 1
                gd[5 * i + 1] = candle_charecteristic_values[0]
                gd[5*i + 4] = candle_charecteristic_values[1]
                gd[5*i + 2] = lag_values[0]
                gd[5*i + 5] = lag_values[1]

                    # No comparison between the same charecteristic of a candle for the same lag
                if gd[5 * i + 1] == gd[5*i + 4] and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0


                 # We can't compare the high and low of the same candle
                elif (gd[5*i + 1] == 'High' and gd[5 * i + 4] == 'Low') or (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0
        

                # No high against open for the same candle
                elif (gd[5*i + 1] ==  'High' and gd[5 * i + 4] == 'Open') or (gd[5*i + 1] == 'Open' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0

        
                # No high against close for the same candle
                elif (gd[5*i + 1] == 'High' and gd[5 * i + 4] == 'Close') or (gd[5*i + 1] == 'Close' and gd[5 * i + 4] == 'High') and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0
        

                # No low against open for the same candle
                elif (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'Open') or (gd[5*i + 1] == 'Open' and gd[5 * i + 4] == 'Low') and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0


                # No low against close for the same candle
                elif (gd[5*i + 1] == 'Low' and gd[5 * i + 4] == 'Close') or (gd[5*i + 1] == 'Close' and gd[5 * i + 4] == 'Low') and gd[5*i + 2] == gd[5*i + 5]:
                    robust = 0    

                # No open against previous candle high, first variaton
                elif (gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == -1):
                    robust = 0

        
                # No open against previous candle high, second variation
                elif (gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1):
                    robust = 0


                # No open against previous candle low, first variation
                elif gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == -1:
                    robust = 0


                # No open against previous candle low, second variation
                elif gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1:
                    robust = 0


                # No open against previous candle close, first variation
                elif gd[5*i + 1] == 'Open' and gd[5*i + 4] == 'Close' and gd[5*i + 2] - gd[5*i + 5] == -1:
                    robust = 0


                # No open against previous candle close, second varion
                elif gd[5*i + 1] == 'Close' and gd[5*i + 4] == 'Open' and gd[5*i + 2] - gd[5*i + 5] == 1:
                    robust = 0


                # No high of current gainst low of previous candle, first variation
                elif gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == -1:
                    robust = 0


                # No high of cuurent candle against low of previous candle, second variation        
                elif gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == 1:
                    robust = 0  

                # No low of current candle against high of previous candle, first variation   
                elif gd[5*i + 1] == 'Low' and gd[5*i + 4] == 'High' and gd[5*i + 2] - gd[5*i + 5] == -1:
                    robust = 0

                # No low of current candle against high of previous candle, second variation   
                elif gd[5*i + 1] == 'High' and gd[5*i + 4] == 'Low' and gd[5*i + 2] - gd[5*i + 5] == 1:
                    robust = 0
                
                else:
                    robust = 1

    # Returns the strategy insuring it is correctly specified
    return gd



def get_fitness_function(gd):

    # The two variables needed to calculate the Ulcer Index, sumsq is the squred sum of DrawDown and Max Value is the highest value that the account reached
    sumsq = 0
    max_value = 0
    value = 100

    # We start by creating an adaptive condition that will be used to calculate the log return
    # Also instead of writing generation_dictionary I am  abbreviating it to gd

    # This creates an array matching the length of the training set
    condition = np.ones(len(train_df), dtype=bool)


    # This works, returns an array for each candle if a candle was confirmed

    for i in range(pattern_size):
        condition &= gd[5* i + 3](train_df[gd[5*i + 1]].shift(gd[5*i + 2]), train_df[gd[5*i + 4]].shift(gd[5 * i + 5]))

    if longs == 1:
        results = np.where(condition, np.log( train_df['Close'].shift(-1)/ train_df['Close']), 0)
        results = np.nan_to_num(results, nan = 0)
    else:
        results = np.where(condition, - np.log(train_df['Close'] / train_df['Close'].shift(-1)), 0)
        results = np.nan_to_num(results, nan = 0)      


    # This snippet calculates the Ulcer Index, which will be used as the denominator of the Martin Ratio, which is the
    # fitness function this algorithm will use
    #-------------That is the general Idea behind the Calculation of the Ulcer Index
    #SumSq = 0
    #MaxValue = 0
    #for T = 1 to NumOfPeriods do
    #if Value[T] > MaxValue then MaxValue = Value[T]
    #else SumSq = SumSq + sqr(100 * ((Value[T] / MaxValue) - 1))
    #UI = sqrt(SumSq / NumOfPeriods)

    # If trades are less than 2.5% (minimum frequency) of the candles automatically the fitness function is set to 0
    if condition.sum() < minimum_frequency * len(condition):
        martin_ratio = 0

    else:
        for elem in results:
            value = value  + value * elem

            if value >= max_value:
                max_value = value
            else:
                sumsq = sumsq + (100 * ((value/max_value) - 1))*(100 * ((value/max_value) - 1))
        
            ulcer_index = math.sqrt(sumsq/len(results))

        martin_ratio = value / ulcer_index

    # Also if the martion ratio is negative, it is set to 0:
    if martin_ratio < 0:
        martin_ratio = 0

    global best_performer
    if martin_ratio >= best_performer[0][5*pattern_size + 1]:
        best_performer[0] = gd
        global return_distribution_best_performer
        return_distribution_best_performer = results

    

    return martin_ratio

def get_fitness_function_subsequent_run(gd, all_return_distributions = all_return_distributions):

    # The two variables needed to calculate the Ulcer Index, sumsq is the squred sum of DrawDown and Max Value is the highest value that the account reached
    sumsq = 0
    max_value = 0
    value = 100

    correlation_list = []

    condition = np.ones(len(train_df), dtype=bool)

    for i in range(pattern_size):
        condition &= gd[5* i + 3](train_df[gd[5*i + 1]].shift(gd[5*i + 2]), train_df[gd[5*i + 4]].shift(gd[5 * i + 5]))


    if longs == 1:
        results = np.where(condition, np.log(train_df['Close'].shift(-1)/ train_df['Close']), 0)
        results = np.nan_to_num(results, nan = 0)
    else:
        results = np.where(condition, - np.log(train_df['Close'] / train_df['Close'].shift(-1)), 0)
        results = np.nan_to_num(results, nan = 0)    

    #Explanations in the previous function
    if condition.sum() < minimum_frequency * len(condition):
        martin_ratio = 0

    else:
        for elem in results:
            value = value  + value * elem

            if value >= max_value:
                max_value = value
            else:
                sumsq = sumsq + (100 * ((value/max_value) - 1))*(100 * ((value/max_value) - 1))
        
            ulcer_index = math.sqrt(sumsq/len(results))

        martin_ratio = value / ulcer_index


    #Calculates all correlation coefficients between the best performers and the current return distribution
    for i in range(len(all_return_distributions)):
        correlation_matrix = np.corrcoef(all_return_distributions[i], results)
        correlation = correlation_matrix[0, 1]
        correlation_list.append(correlation)


    max_correlation = max(correlation_list)
    if np.isnan(max_correlation):
        max_correlation = 0
    martin_ratio = (1 - max_correlation) * martin_ratio

    # Also if the martion ratio is negative, it is set to 0:
    if martin_ratio < 0:
        martin_ratio = 0

    global best_performer
    if martin_ratio >= best_performer[len(all_return_distributions)][5*pattern_size + 1]:
        best_performer[len(all_return_distributions)] = gd
        global return_distribution_best_performer
        return_distribution_best_performer = results

    

    return martin_ratio


def mutate_gene(gd):

    candle_charecteristic_keys = get_candle_charesteristic_keys()
    lag_keys = [elem + 1 for elem in candle_charecteristic_keys]
    
    # This generates 1 or 0, which decides whether we will mutatue a lag or candle charecteristic
    rand = random.randint(0 ,1)

    # If rand is 0, we mutate a candle charecteristic
    if rand == 0:
        key = random.choice(candle_charecteristic_keys)
        possible_candle_value = [value for value in candle_genes.values() if value != gd[key]]
        gd[key] = random.choice(possible_candle_value)

    # If rand is 1, we mutate a lag value
    if rand == 1:
        key = random.choice(lag_keys)
        possible_lag_value = [value for value in lag_genes.values() if value != gd[key]]
        gd[key] = random.choice(possible_lag_value)

    return gd
        
def mutate_pattern(gd):
    # This generates at entirely new random pattern
    candle_charecteristic_keys = get_candle_charesteristic_keys()
    lag_keys = [elem + 1 for elem in candle_charecteristic_keys]
    comparison_keys = [lag_keys[i] + 1 for i in range (0, len(lag_keys), 2)]



    candle_charecteristic_values = np.random.choice(list(candle_genes.values()), size=len(candle_charecteristic_keys), replace=True)
    lag_values = np.random.choice(list(lag_genes.values()), size = len(lag_keys), replace = True)
    comparison_values = np.random.choice(list(comparison_genes.values()), size = len(comparison_keys), replace = True)

    for i in range(1 ,5 * pattern_size + 1, 1):

        if i in candle_charecteristic_keys:
            gd[i] = candle_charecteristic_values[candle_charecteristic_keys.index(i)]
        elif i in lag_keys:
            gd[i] = lag_values[lag_keys.index(i)]
        elif i in comparison_keys:
            gd[i] = comparison_values[comparison_keys.index(i)]
    
    gd = strategy_robustness_check(gd)
    return gd

    
    


# Will seed the first generation and return a sorted list of dictionaries with all the strategies
def seed_first_generation():
    # There are 3 random assignment operations that need to be done
    # - Type of candle charecteristic - O, H, L , C
    # - Lag - 0 , 1 , 2, 3 etc..
    # - Comparison - >, <

    # Here we define th places where the genes for candle stick charecteristics, lags and comparison need to be implemented
    candle_charecteristic_keys = get_candle_charesteristic_keys()
    lag_keys = [elem + 1 for elem in candle_charecteristic_keys]
    comparison_keys = [lag_keys[i] + 1 for i in range (0, len(lag_keys), 2)]



    candle_charecteristic_values = [np.random.choice(list(candle_genes.values()), size=len(candle_charecteristic_keys), replace=True) for i in range(0, population_size)]
    lag_values = [np.random.choice(list(lag_genes.values()), size = len(lag_keys), replace = True) for i in range(0, population_size)]
    comparison_values = [np.random.choice(list(comparison_genes.values()), size = len(comparison_keys), replace = True) for i in range(0, population_size)]


    first_generation = create_generation()

    # This insures that all the keys in the dictionary are assigned the proper values and creates the first generation seed with actual values
    # It creates 200 chromozomes in this case up to the population size
    for j in range(0, population_size):
        for i in range(1 ,5 * pattern_size + 1, 1):

            if i in candle_charecteristic_keys:
                first_generation[j][i] = candle_charecteristic_values[j][candle_charecteristic_keys.index(i)]
            elif i in lag_keys:
                first_generation[j][i] = lag_values[j][lag_keys.index(i)]
            elif i in comparison_keys:
                first_generation[j][i] = comparison_values[j][comparison_keys.index(i)]
        
        # After each strategy is generated we call the robust function to check that it is indeed robust and correct it if needed
        first_generation[j] = strategy_robustness_check(first_generation[j])
        # Finally we compute the fitness function

        first_generation[j][5 * pattern_size + 1] = get_fitness_function(first_generation[j])


        
    # We sort the generation so that first is the strategy with the highest value of the fitness function and so on in descending order
    first_generation = sorted(first_generation, key = lambda x: x[5*pattern_size + 1], reverse = True)
    return first_generation

def seed_first_generation_subsequent_run():
    # There are 3 random assignment operations that need to be done
    # - Type of candle charecteristic - O, H, L , C
    # - Lag - 0 , 1 , 2, 3 etc..
    # - Comparison - >, <

    # Here we define th places where the genes for candle stick charecteristics, lags and comparison need to be implemented
    candle_charecteristic_keys = get_candle_charesteristic_keys()
    lag_keys = [elem + 1 for elem in candle_charecteristic_keys]
    comparison_keys = [lag_keys[i] + 1 for i in range (0, len(lag_keys), 2)]



    candle_charecteristic_values = [np.random.choice(list(candle_genes.values()), size=len(candle_charecteristic_keys), replace=True) for i in range(0, population_size)]
    lag_values = [np.random.choice(list(lag_genes.values()), size = len(lag_keys), replace = True) for i in range(0, population_size)]
    comparison_values = [np.random.choice(list(comparison_genes.values()), size = len(comparison_keys), replace = True) for i in range(0, population_size)]


    first_generation = create_generation()

    # This insures that all the keys in the dictionary are assigned the proper values and creates the first generation seed with actual values
    # It creates 200 chromozomes in this case up to the population size
    for j in range(0, population_size):
        for i in range(1 ,5 * pattern_size + 1, 1):

            if i in candle_charecteristic_keys:
                first_generation[j][i] = candle_charecteristic_values[j][candle_charecteristic_keys.index(i)]
            elif i in lag_keys:
                first_generation[j][i] = lag_values[j][lag_keys.index(i)]
            elif i in comparison_keys:
                first_generation[j][i] = comparison_values[j][comparison_keys.index(i)]
        
        # After each strategy is generated we call the robust function to check that it is indeed robust and correct it if needed
        first_generation[j] = strategy_robustness_check(first_generation[j])
        # Finally we compute the fitness function

        first_generation[j][5 * pattern_size + 1] = get_fitness_function_subsequent_run(first_generation[j])


        
    # We sort the generation so that first is the strategy with the highest value of the fitness function and so on in descending order
    first_generation = sorted(first_generation, key = lambda x: x[5*pattern_size + 1], reverse = True)
    return first_generation
        

def run_algorithm_first_itteration(elitism = elitism, mutation_chance = mutation_chance, fresh_pattern_chance = fresh_pattern_chance, max_generation_number = max_generation_number ):



    return_distribution_best_perfomer = []

    # We seed the first generation
    first_generation = seed_first_generation()


    # Create an empty array that will hold information about all the generations
    generation_array = []

    generation_array.append(first_generation)


    and_position_array = get_and_positions()

    for i in range(max_generation_number):
            
        subsequent_generation = []

            #Based on the value of elitism the first n=highest performing strategies will be inhereted to the new population
            # Elitism protects patterns from mutation in the pattern and element form
        for j in range(0,elitism):
            subsequent_generation.append(generation_array[i][j])

            # Now it needs to loop over the rest of the generation to create new strategies, by parent crossing
            # with a chance of gene or pattern mutation, probabilities will be applied the following way:
            # They apply their fixed probability once for each strategy, so in this case 5% gene mutation chance chosen randomly if true
            #and 2% for a fresh pattern chance again the same logic

        for j in range(elitism, math.ceil(population_size/2), 1):

                # We take care of the parent selection, by summing all the fitness functions together and then generating a random number
                # from 0 to the sum, then from the first and onwards we calculate the cummilative sum, the first strategy for which the cumilative 
                # function exceeds the random number, is the first parent same idea for the second, then they create 2 children, by selecting a a random
                # comparison cut-off point, the first child inherits the logic from the first parrent up until the cutoff point and the rest from the second
                # parent after the cutoff point, vice versa for the second child

            total_fitness_function_sum = sum(d[5 * pattern_size + 1] for d in generation_array[i])




                # Assigns the First parent
            sum_first_parent = random.randint(0, round(total_fitness_function_sum))
            cumilative_sum_first_parent = 0
            first_parent = {}


            while cumilative_sum_first_parent < sum_first_parent:
                for m in range(population_size):
                    most_recent = generation_array[i][m]
                    cumilative_sum_first_parent += generation_array[i][m][5* pattern_size + 1]

                    if cumilative_sum_first_parent >= sum_first_parent:
                        first_parent = most_recent
                        break
                
                

                # Assigns the second parent based on the logic
            sum_second_parent = random.randint(0, round(total_fitness_function_sum))
            cumilative_sum_second_parent = 0
            second_parent = {}

            while cumilative_sum_second_parent < sum_second_parent:
                for m in range(population_size):
                    most_recent = generation_array[i][m]
                    cumilative_sum_second_parent += generation_array[i][m][5* pattern_size + 1]

                    if cumilative_sum_second_parent >= sum_second_parent:
                        second_parent = most_recent
                        break
                                  



            current_cutoff_point = random.choice(and_position_array)

            try:
                first_child = {**{k: first_parent[k] for k in range(current_cutoff_point)}, **{m: second_parent[m] for m in range(current_cutoff_point, 5 * pattern_size + 1)}}
                second_child = {**{k: second_parent[k] for k in range(current_cutoff_point)}, **{m: first_parent[m] for m in range(current_cutoff_point, 5 * pattern_size + 1)}}
            except:
                pass


            # Generate the 2 random numbers that will be used for gene mutation
            rand_num_1_first_child = round(random.uniform(0, 100),2) / 100
            rand_num_1_second_child = round(random.uniform(0, 100),2) / 100

            # Generate the second random number, that will be used for pattern mutatuin
            rand_num_2_first_child = round(random.uniform(0, 100),2) / 100
            rand_num_2_second_child = round(random.uniform(0, 100),2) / 100



            #Take care of gene mutation, it is important to note that genes which are useless can evolve,
            #i.e. strategies which lose their profitability due to a condition that becomes impossible

            if rand_num_1_first_child <= mutation_chance:
                
                first_child = mutate_gene(first_child)
                #print('First Child Gene Mutation')

            if rand_num_1_second_child <= mutation_chance:
                
                second_child = mutate_gene(second_child)
                #print('Second Child Gene Mutation')

                # Mutates a pattern by generating an entirely new one
            if rand_num_2_first_child <= fresh_pattern_chance:
                first_child = mutate_pattern(first_child)
                #print('First Child Pattern Mutation')

            if rand_num_2_second_child <= fresh_pattern_chance:
                second_child = mutate_pattern(second_child)
                #print('Second Child PAttern Mutation')

                
                # Now calculate the fitness function of the children

            first_child[5*pattern_size + 1] = np.nan
            second_child[5*pattern_size + 1] = np.nan


            first_child[5 * pattern_size + 1] = get_fitness_function(first_child)
            second_child[5 * pattern_size + 1] = get_fitness_function(second_child)



            subsequent_generation.append(first_child)
            subsequent_generation.append(second_child)
        
        # Orders the generation is descending order based on the total fitness function and appends it to the generation array
        subsequent_generation = sorted(subsequent_generation, key = lambda x: x[5*pattern_size + 1], reverse = True)
        generation_array.append(subsequent_generation)

        fitness_functions_mean = sum([d[5 * pattern_size + 1] for d in subsequent_generation])/len(subsequent_generation)
        max_fitness_function_current_generation = subsequent_generation[0][5*pattern_size+ 1]
        #print(f"This is the {i + 2} generation, mean fitness: {fitness_functions_mean}, max: {max_fitness_function_current_generation}")


        global best_performer
        #print(f"According to best_performer: {best_performer}")
        #print(f"According to script: {subsequent_generation[0]}")
    global return_distribution_best_performer, all_return_distributions
    all_return_distributions.append(return_distribution_best_performer)
    print(len(all_return_distributions))
    best_performer[0] = generation_array[max_generation_number][0]
    return best_performer[0]

def run_algorithm_subsequent_itteration(elitism = elitism, mutation_chance = mutation_chance, fresh_pattern_chance = fresh_pattern_chance, max_generation_number = max_generation_number, total_runs = total_runs ):

    # We seed the first generation
    first_generation = seed_first_generation_subsequent_run()

    # Create an empty array that will hold information about all the generations
    generation_array = []

    generation_array.append(first_generation)


    and_position_array = get_and_positions()

    for i in range(max_generation_number):
            
        subsequent_generation = []

            #Based on the value of elitism the first n=highest performing strategies will be inhereted to the new population
            # Elitism protects patterns from mutation in the pattern and element form
        for j in range(0,elitism):
            subsequent_generation.append(generation_array[i][j])

            # Now it needs to loop over the rest of the generation to create new strategies, by parent crossing
            # with a chance of gene or pattern mutation, probabilities will be applied the following way:
            # They apply their fixed probability once for each strategy, so in this case 5% gene mutation chance chosen randomly if true
            #and 2% for a fresh pattern chance again the same logic

        for j in range(elitism, math.floor(population_size/2), 1):

                # We take care of the parent selection, by summing all the fitness functions together and then generating a random number
                # from 0 to the sum, then from the first and onwards we calculate the cummilative sum, the first strategy for which the cumilative 
                # function exceeds the random number, is the first parent same idea for the second, then they create 2 children, by selecting a a random
                # comparison cut-off point, the first child inherits the logic from the first parrent up until the cutoff point and the rest from the second
                # parent after the cutoff point, vice versa for the second child

            total_fitness_function_sum = sum(d[5 * pattern_size + 1] for d in generation_array[i])

                # Assigns the First parent
            sum_first_parent = random.randint(0, round(total_fitness_function_sum))
            cumilative_sum_first_parent = 0
            first_parent = {}


            while cumilative_sum_first_parent < sum_first_parent:
                for m in range(population_size):
                    try:
                        most_recent = generation_array[i][m]
                        cumilative_sum_first_parent += generation_array[i][m][5* pattern_size + 1]
                    except:
                        print(f"i is {i}, m is {m}")
                        

                    if cumilative_sum_first_parent >= sum_first_parent:
                        first_parent = most_recent
                        break
                
                

                # Assigns the second parent based on the logic
            sum_second_parent = random.randint(0, round(total_fitness_function_sum))
            cumilative_sum_second_parent = 0
            second_parent = {}

            while cumilative_sum_second_parent < sum_second_parent:
                for m in range(population_size):
                    try:
                        most_recent = generation_array[i][m]
                        cumilative_sum_second_parent += generation_array[i][m][5* pattern_size + 1]
                    except:
                        print(f"i is {i}, m is {m}")

                    if cumilative_sum_second_parent >= sum_second_parent:
                        second_parent = most_recent
                        break
                                  



            current_cutoff_point = random.choice(and_position_array)

            try:
                first_child = {**{k: first_parent[k] for k in range(current_cutoff_point)}, **{m: second_parent[m] for m in range(current_cutoff_point, 5 * pattern_size + 1)}}
                second_child = {**{k: second_parent[k] for k in range(current_cutoff_point)}, **{m: first_parent[m] for m in range(current_cutoff_point, 5 * pattern_size + 1)}}
            except:
                pass



            # Generate the 2 random numbers that will be used for gene mutation
            rand_num_1_first_child = round(random.uniform(0, 100),2) / 100
            rand_num_1_second_child = round(random.uniform(0, 100),2) / 100

            # Generate the second random number, that will be used for pattern mutatuin
            rand_num_2_first_child = round(random.uniform(0, 100),2) / 100
            rand_num_2_second_child = round(random.uniform(0, 100),2) / 100



            #Take care of gene mutation, it is important to note that genes which are useless can evolve,
            #i.e. strategies which lose their profitability due to a condition that becomes impossible

            if rand_num_1_first_child <= mutation_chance:
                
                first_child = mutate_gene(first_child)
                #print('First Child Gene Mutation')

            if rand_num_1_second_child <= mutation_chance:
                
                second_child = mutate_gene(second_child)
                #print('Second Child Gene Mutation')

                # Mutates a pattern by generating an entirely new one
            if rand_num_2_first_child <= fresh_pattern_chance:
                first_child = mutate_pattern(first_child)
                #print('First Child Pattern Mutation')

            if rand_num_2_second_child <= fresh_pattern_chance:
                second_child = mutate_pattern(second_child)
                #print('Second Child PAttern Mutation')

                
                # Now calculate the fitness function of the children

            first_child[5*pattern_size + 1] = np.nan
            second_child[5*pattern_size + 1] = np.nan


            first_child[5 * pattern_size + 1] = get_fitness_function_subsequent_run(first_child)
            second_child[5 * pattern_size + 1] = get_fitness_function_subsequent_run(second_child)



            subsequent_generation.append(first_child)
            subsequent_generation.append(second_child)
        
        # Orders the generation is descending order based on the total fitness function and appends it to the generation array
        subsequent_generation = sorted(subsequent_generation, key = lambda x: x[5*pattern_size + 1], reverse = True)
        generation_array.append(subsequent_generation)

        fitness_functions_mean = sum([d[5 * pattern_size + 1] for d in subsequent_generation])/len(subsequent_generation)
        max_fitness_function_current_generation = subsequent_generation[0][5*pattern_size+ 1]
        #print(f"This is the {i + 2} generation, mean fitness: {fitness_functions_mean}, max: {max_fitness_function_current_generation}")


        global best_performer
        #print(f"According to best_performer: {best_performer}")
        #print(f"According to script: {subsequent_generation[0]}")
    global return_distribution_best_performer, all_return_distributions
    all_return_distributions.append(return_distribution_best_performer)
    return_distribution_best_performer = []
    best_performer[len(all_return_distributions) - 1] = generation_array[max_generation_number][0]
    return best_performer[len(all_return_distributions) - 1]

                

def run_genetic_algorithm():
    first_pattern = run_algorithm_first_itteration()
    #print(first_pattern)
    subsuquent_patterns = [run_algorithm_subsequent_itteration() for i in range(1, total_runs)]
#print(subsuquent_patterns)

    patterns = [ elem for elem in subsuquent_patterns]
    patterns.append(first_pattern)
    #print(patterns)

    
#[{0: None, 1: 'Close', 2: 1, 3: <built-in function gt>, 4: 'High', 5: 3, 6: 'Close', 7: 0, 8: <built-in function gt>, 9: 'Open', 10: 0, 11: 'High', 12: 2, 13: <built-in function lt>, 14: 'High', 15: 3, 16: 251.28276931215996}, {0: None, 1: 'Close', 2: 3, 3: <built-in function lt>, 4: 'Low', 5: 2, 6: 'Open', 7: 3, 8: <built-in function lt>, 9: 'Open', 10: 1, 11: 'High', 12: 0, 13: <built-in function gt>, 14: 'High', 15: 2, 16: 157.95846858793323}, {0: None, 1: 'Open', 2: 1, 3: <built-in function gt>, 4: 'Low', 5: 0, 6: 'Open', 7: 3, 8: <built-in function gt>, 9: 'High', 10: 1, 11: 'Open', 12: 1, 13: <built-in function gt>, 14: 'Close', 15: 3, 16: 125.61590144849566}, {0: None, 1: 'Close', 2: 2, 3: <built-in function gt>, 4: 'Close', 5: 0, 6: 'Open', 7: 3, 8: <built-in function lt>, 9: 'Close', 10: 0, 11: 'High', 12: 1, 13: <built-in function lt>, 14: 'High', 15: 0, 16: 108.55779572693544}]
#[{0: None, 1: 'Close', 2: 3, 3: <built-in function lt>, 4: 'Low', 5: 0, 6: 'Close', 7: 1, 8: <built-in function gt>, 9: 'Open', 10: 2, 11: 'Low', 12: 2, 13: <built-in function gt>, 14: 'Low', 15: 1, 16: 119.79694773154492}, {0: None, 1: 'Low', 2: 2, 3: <built-in function gt>, 4: 'High', 5: 3, 6: 'Close', 7: 3, 8: <built-in function lt>, 9: 'Low', 10: 2, 11: 'Close', 12: 0, 13: <built-in function gt>, 14: 'Low', 15: 1, 16: 191.25956310593747}, {0: None, 1: 'Open', 2: 0, 3: <built-in function lt>, 4: 'Open', 5: 1, 6: 'Close', 7: 1, 8: <built-in function lt>, 9: 'High', 10: 3, 11: 'Close', 12: 0, 13: <built-in function gt>, 14: 'Open', 15: 1, 16: 121.34313081575395}, {0: None, 1: 'Close', 2: 0, 3: <built-in function lt>, 4: 'High', 5: 2, 6: 'Open', 7: 3, 8: <built-in function lt>, 9: 'Low', 10: 1, 11: 'High', 12: 0, 13: <built-in function gt>, 14: 'High', 15: 2, 16: 112.32356651018164}, {0: None, 1: 'Open', 2: 3, 3: <built-in function gt>, 4: 'High', 5: 1, 6: 'Open', 7: 1, 8: <built-in function gt>, 9: 'Close', 10: 3, 11: 'Open', 12: 2, 13: <built-in function gt>, 14: 'Low', 15: 3, 16: 154.1051397783713}]
    for j in range(len(patterns)):

        condition = np.ones(len(train_df), dtype=bool)

    # This works, returns an array for each candle if a candle was confirmed
        for i in range(pattern_size):
            condition &= patterns[j][5* i + 3](train_df[patterns[j][5*i + 1]].shift(patterns[j][5*i + 2]), train_df[patterns[j][5*i + 4]].shift(patterns[j][5 * i + 5]))


        results = np.where(condition, np.log(train_df['Close'].shift(-1)/ train_df['Close']), 0)
        results = np.nan_to_num(results, nan = 0)

        cumilative_return = np.sum(results)
        profit_factor = np.sum(- results[ results > 0]) / np.sum(results[results < 0])
        time_in_market = np.count_nonzero(results) / len(results) * 100

        #print(f"Pattern number {j}, cumilative return: {cumilative_return}, Profit Factor: {profit_factor}, Time in the market {time_in_market} %")
        #print(patterns[j])

    return patterns

def out_of_sample_test(patterns):
    print("Out of Sample Test results")

    returns = 1

    for j in range(len(patterns)):

        condition = np.ones(len(test_df), dtype=bool)

    # This works, returns an array for each candle if a candle was confirmed
        for i in range(pattern_size):
            condition &= patterns[j][5* i + 3](test_df[patterns[j][5*i + 1]].shift(patterns[j][5*i + 2]), test_df[patterns[j][5*i + 4]].shift(patterns[j][5 * i + 5]))


        results = np.where(condition, np.log(test_df['Close'].shift(-1)/ test_df['Close']), 0)
        results = np.nan_to_num(results, nan = 0)

        cumilative_return = np.sum(results)
        returns= returns * (1 + cumilative_return)
        profit_factor = np.sum(- results[ results > 0]) / np.sum(results[results < 0])
        time_in_market = np.count_nonzero(results) / len(results) * 100

        print(f"Pattern number {j + 1}, cumilative return: {cumilative_return}, Profit Factor: {profit_factor}, Time in the market {time_in_market} %")
        print(patterns[j])
    
    return returns
# Longs

return_array = []
for i in range(5, 10):
    start = round((i -1)/ 10 * len(df))
    end = round( (i+0.5) /10 * len(df))
    end_test = round((i+1.5)/10 * len(df))
    train_df = df[start: end]
    test_df = df[end : end_test]
    long_patterns = run_genetic_algorithm()
    return_array.append(out_of_sample_test(long_patterns) - 1)
    print(f"Walk forward test returns are: {return_array}")
