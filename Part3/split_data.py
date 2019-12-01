import numpy as np
import pandas as pd

# Using splitting technique recommended here: 
# https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing

FRACTION_TRAIN = 0.85

# Read the csv file
data = pd.read_csv("heart.csv")
rng = np.random.RandomState()

# Split data into a training set and testing set
train_data = data.sample(frac=FRACTION_TRAIN, random_state=rng)
test_data = data.loc[~data.index.isin(train_data.index)]

# Save the training and testing data to a new csv file
train_data.to_csv("heart_train.csv", index=False)
test_data.to_csv("heart_test.csv", index=False)