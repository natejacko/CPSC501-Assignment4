import tensorflow as tf
import numpy as np
import pandas as pd
import functools

# Numerical preprocessor to pack list of numerical features into single column
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels

def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std

print("--Get data--")
# Path to csv files
TRAIN_DATA_PATH = "heart_train.csv"
TEST_DATA_PATH = "heart_test.csv"

# Label of column to predicted
LABEL_COLUMN = "chd"

# Loading csv files as datasets
BATCH_SIZE = 5

raw_train_dataset = tf.data.experimental.make_csv_dataset(TRAIN_DATA_PATH, batch_size=BATCH_SIZE, label_name=LABEL_COLUMN)
raw_test_dataset = tf.data.experimental.make_csv_dataset(TEST_DATA_PATH, batch_size=BATCH_SIZE, label_name=LABEL_COLUMN)

print("--Process data--")
# Labels of columns containing numerical data
NUMERIC_FEATURES = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]

# Pack the numerical columns of the training and testing data into a single column
packed_train_data = raw_train_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
x_train, y_train = next(iter(packed_train_data))

packed_test_data = raw_test_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
x_test, y_test = next(iter(packed_test_data)) 

# Normalize the numerical data
desc = pd.read_csv(TRAIN_DATA_PATH)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

# Create numerical columns
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

# Labels of columns containing categorical data (and their categories)
CATEGORIES = {"famhist": ["Present", "Absent"]}

# Create categorical columns
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

print("--Make model--")
# Combine numerical and categorical columns into a single layer
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

# Build the Keras model
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

STEPS_PER_EPOCH = 128

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(packed_train_data, epochs=20, steps_per_epoch=STEPS_PER_EPOCH)

print("--Evaluate model--")
# Evaluate the model on test data
model_loss, model_acc = model.evaluate(packed_test_data, steps=STEPS_PER_EPOCH)

print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%") 