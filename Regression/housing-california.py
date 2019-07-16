import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Load Housing dataset
HOUSING_PATH = "Regression\Datasets"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
# housing.head()
# housing.info()
# housing.describe()


# Plot histograms
def plot_hist():
    housing.hist(bins=50, figsize=(10, 7))
    plt.show()

plot_hist()


# Split data into train and test sets
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("Train: ", len(train_set), "+ Test: ", len(test_set))


# housing.hist(column='median_income', bins=10)
# plt.show()