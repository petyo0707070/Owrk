################################################################################
# 1) FX_Hedging

library(readxl)
library(dplyr)


rm(list = ls())

# 1. Load data
file_path <- "/Users/nedimkorcic/Documents/Quantitative Finance Master/Asset & Risk Management 1/Final/Shortfall (Risk Ex)/riskmanagementexample_data.xlsx"
data <- read_excel(file_path, sheet = "data")

# 2. Basic conversions and returns
data <- data %>%
  mutate(
    # Ensure date is in proper Date format
    date = as.Date(date),
    # Convert USD/EUR to EUR/USD (Excel: =1/C2)
    EUR_per_USD = 1 / USD_per_EUR_PI,
    
    # Calculate simple returns (Excel: =B3/B2-1, etc.)
    R_spx_USD = (SPX_TRI / lag(SPX_TRI)) - 1,
    r_EURUSD = (EUR_per_USD / lag(EUR_per_USD)) - 1,
    R_gov_USD = (USGOV_TRI / lag(USGOV_TRI)) - 1,
    R_omv_EUR = (OMV_TRI / lag(OMV_TRI)) - 1,
    
    # FX-adjusted returns (Excel: =(1+B3)*(1+D3)-1)
    R_spx_EUR = (1 + R_spx_USD) * (1 + r_EURUSD) - 1,
    R_gov_EUR = (1 + R_gov_USD) * (1 + r_EURUSD) - 1
  ) %>%
  na.omit()  # Remove rows with missing values

# 3. Hedge ratios for S&P 500 (Full sample)
spx_model <- lm(R_spx_EUR ~ r_EURUSD, data = data)
h_star_spx <- coef(spx_model)[2]
print(paste("S&P 500 optimal hedge ratio (full sample):", round(h_star_spx, 4)))

# 4. Gedge ratios for US bonds (Full sample)
gov_model <- lm(R_gov_EUR ~ r_EURUSD, data = data)
h_star_gov <- coef(gov_model)[2]
print(paste("US bonds optimal hedge ratio (full sample):", round(h_star_gov, 4)))

# 5. Gedge ratios for OMV (Full sample)
omv_model <- lm(R_omv_EUR ~ r_EURUSD, data = data)
h_star_omv <- coef(omv_model)[2]
print(paste("OMV optimal hedge ratio (full sample):", round(h_star_omv, 4)))

# 6. Period-specific analysis (Pre-2020)
data_pre2020 <- data %>% filter(date < as.Date("2020-01-01"))
spx_model_pre2020 <- lm(R_spx_EUR ~ r_EURUSD, data = data_pre2020)
print(paste("S&P 500 h* pre-2020:", round(coef(spx_model_pre2020)[2], 4)))

# 7. Period-specific analysis (2020)
data_2020 <- data %>% filter(date >= as.Date("2020-01-01") & date < as.Date("2021-01-01"))
spx_model_2020 <- lm(R_spx_EUR ~ r_EURUSD, data = data_2020)
print(paste("S&P 500 h* for 2020:", round(coef(spx_model_2020)[2], 4)))

# 8. Period-specific analysis (Post-2020)
data_post2020 <- data %>% filter(date >= as.Date("2021-01-01"))
spx_model_post2020 <- lm(R_spx_EUR ~ r_EURUSD, data = data_post2020)
print(paste("S&P 500 h* post-2020:", round(coef(spx_model_post2020)[2], 4)))

# 9. Add optimally hedged returns to main data
data <- data %>%
  mutate(
    R_spx_opt_hedged = R_spx_EUR - h_star_spx * r_EURUSD,
    R_gov_opt_hedged = R_gov_EUR - h_star_gov * r_EURUSD,
    R_omv_opt_hedged = R_omv_EUR - h_star_omv * r_EURUSD
  )

# 10. Show first few rows of final data
print("First few rows of final dataset:")
head(data) 






################################################################################
## 2) VaR Historical Simulation


rm(list = ls())

#_______________________________________________________________________________
# Copied from Assignment 1
# 1. Clear workspace and load data
file_path <- "/Users/nedimkorcic/Documents/Quantitative Finance Master/Asset & Risk Management 1/Final/Shortfall (Risk Ex)/riskmanagementexample_data.xlsx"
data <- read_excel(file_path, sheet = "data")

# 2. Calculate basic conversions and returns
data <- data %>%
  mutate(
    # Ensure date is in proper Date format
    date = as.Date(date),
    # Convert USD/EUR to EUR/USD (Excel: =1/C2)
    EUR_per_USD = 1 / USD_per_EUR_PI,
    
    # Calculate simple returns (Excel: =B3/B2-1, etc.)
    R_spx_USD = (SPX_TRI / lag(SPX_TRI)) - 1,
    r_EURUSD = (EUR_per_USD / lag(EUR_per_USD)) - 1,
    R_gov_USD = (USGOV_TRI / lag(USGOV_TRI)) - 1,
    R_omv_EUR = (OMV_TRI / lag(OMV_TRI)) - 1,
    
    # FX-adjusted returns (Excel: =(1+B3)*(1+D3)-1)
    R_spx_EUR = (1 + R_spx_USD) * (1 + r_EURUSD) - 1,
    R_gov_EUR = (1 + R_gov_USD) * (1 + r_EURUSD) - 1
  ) %>%
  na.omit()  # Remove rows with missing values
#_______________________________________________________________________________





data <- data %>%
  # First, arrange by date to ensure chronological order
  arrange(date) %>%
  # Add week number column at the beginning
  mutate(week = 1:n()) %>%
  # Then calculate 4-week returns
  mutate(
    rolling_4week_fwd_return = sapply(
      1:n(),
      function(i) {
        if (i + 3 <= n()) {
          prod(1 + R_spx_USD[i:(i + 3)]) - 1
        } else {
          NA
        }
      }
    )
  ) %>%
  # Remove incomplete windows
  filter(!is.na(rolling_4week_fwd_return))




#_______________________________________________________________________________
# Mean and St Dev

# Calculate mean and standard deviation
mean_return <- mean(data$R_spx_USD, na.rm = TRUE)  # na.rm removes missing values
sd_return <- sd(data$R_spx_USD, na.rm = TRUE)      # sample standard deviation

# Print mean and SD
cat("Mean weekly return:", mean_return, "\n")
cat("Standard deviation of weekly returns:", sd_return, "\n")

# Compute VaR
var_5 <- qnorm(0.05, mean = mean_return, sd = sd_return)
var_1 <- qnorm(0.01, mean = mean_return, sd = sd_return)

# Print results
cat("5% Delta-Normal VaR:", var_5, "\n")
cat("1% Delta-Normal VaR:", var_1, "\n")
#_______________________________________________________________________________





# Simulation table 
library(dplyr)

# 1. Create the simulation table
sim_table <- data.frame(
  sim = 1:5000,
  zufallszahl = runif(5000)
) %>%
  mutate(
    start_of_4_weeks = ceiling(zufallszahl * 365),
    return = data$rolling_4week_fwd_return[match(start_of_4_weeks, data$week)],
    return = ifelse(is.na(return), 0, return),
    rank = rank(-return, ties.method = "min")
  )

# 2. Add return percentage column
sim_table <- sim_table %>%
  group_by(rank) %>%
  mutate(return_percent = n()/nrow(sim_table)) %>%
  ungroup()

# 3. Calculate quantiles (5% and 1%)
quantile_5pct <- quantile(sim_table$return, 0.05, type = 1)
quantile_1pct <- quantile(sim_table$return, 0.01, type = 1)

# 4. Calculate Expected Shortfall (average return beyond quantile)
expected_shortfall_5pct <- sim_table %>%
  filter(return <= quantile_5pct) %>%
  summarise(ES_5pct = mean(return))

expected_shortfall_1pct <- sim_table %>%
  filter(return <= quantile_1pct) %>%
  summarise(ES_1pct = mean(return))

# View key results
cat("5% Quantile:", quantile_5pct, "\n")
cat("1% Quantile:", quantile_1pct, "\n")
cat("Expected Shortfall (5%):", expected_shortfall_5pct$ES_5pct, "\n")
cat("Expected Shortfall (1%):", expected_shortfall_1pct$ES_1pct, "\n")

