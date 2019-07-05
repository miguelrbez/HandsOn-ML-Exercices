import numpy as np
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import os


# Import MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"]
y = mnist["target"].astype(int)


# Function plots a digit from array of pixels
def plot_digit(digit_array):
    digit_image = digit_array.reshape(28, 28)
    plt.matshow(digit_image, cmap=matplotlib.cm.binary)
    plt.axis("off")
    plt.show()

# plot_digit(X[0])


# Split into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)


# Scales X train and test data
def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)


print("Finished running")