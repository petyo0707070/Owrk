import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a sample DataFrame
data = {
    'A': [10, 20, 50],
    'B': [20, 30, 60],
    'C': [30, 10, 70]
}
df = pd.DataFrame(data)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Calculate the sum across rows (concurrency)
c = df.sum(axis=1)  # Sum of each row
print("\nRow sums (concurrency):")
print(c)

# Step 3: Normalize the DataFrame by dividing each element by the row sum
u = df.div(c, axis=0)
print("\nNormalized DataFrame:")
print(u)

print("New\n")
avgU=u[u>0].mean() # average uniqueness
print(avgU)