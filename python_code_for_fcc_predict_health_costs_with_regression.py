# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

# Data preprocessing and model training

# Examine the dataset
print('Dataset info:')
print(dataset.info())
print('\nDataset description:')
print(dataset.describe())

# Convert categorical data to numerical
# Create a copy of the dataset for processing
processed_dataset = dataset.copy()

# Convert sex: male=1, female=0
processed_dataset['sex'] = processed_dataset['sex'].map({'male': 1, 'female': 0})

# Convert smoker: yes=1, no=0
processed_dataset['smoker'] = processed_dataset['smoker'].map({'yes': 1, 'no': 0})

# Convert region using one-hot encoding
region_dummies = pd.get_dummies(processed_dataset['region'], prefix='region')
processed_dataset = pd.concat([processed_dataset.drop('region', axis=1), region_dummies], axis=1)

print('\nProcessed dataset shape:', processed_dataset.shape)
print('Processed dataset columns:', processed_dataset.columns.tolist())

# Split the data: 80% train, 20% test
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(processed_dataset, test_size=0.2, random_state=42)

# Separate features and labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

print('\nTraining set shape:', train_dataset.shape)
print('Test set shape:', test_dataset.shape)
print('Training labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)

# Normalize the features for better training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_dataset_scaled = scaler.fit_transform(train_dataset)
test_dataset_scaled = scaler.transform(test_dataset)

# Convert to tensorflow datasets
train_dataset = tf.convert_to_tensor(train_dataset_scaled, dtype=tf.float32)
test_dataset = tf.convert_to_tensor(test_dataset_scaled, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

# Build the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[train_dataset.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'mse'])

# Display model architecture
model.summary()

# Train the model
EPOCHS = 200

history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

# Plot training history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [expenses]')
plt.plot(hist['epoch'], hist['mae'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [expenses^2]')
plt.plot(hist['epoch'], hist['mse'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
plt.legend()

plt.tight_layout()
plt.show()

print('\nTraining completed!')

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)

