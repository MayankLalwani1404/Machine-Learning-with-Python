# STEP 1: Fix TensorFlow installation issues in Colab
# Run this cell first, then manually restart runtime (Runtime -> Restart Runtime)

!pip uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu
!pip install --no-cache-dir tensorflow==2.17.0

print('TensorFlow reinstalled. Please restart runtime manually!')
print('Go to Runtime -> Restart Runtime, then run this cell again.')

# STEP 2: After manual runtime restart, uncomment and run these imports:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    print('✓ TensorFlow imported successfully!')
    print('TensorFlow version:', tf.__version__)
except Exception as e:
    print('✗ TensorFlow import failed:', str(e))
    print('Please restart runtime and try again.')

try:
    import tensorflow_datasets as tfds
except ImportError:
    !pip install tensorflow-datasets
    import tensorflow_datasets as tfds

print('All libraries imported successfully!')

# Get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load and explore the data
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

print('Training data shape:', train_data.shape)
print('Test data shape:', test_data.shape)
print('\nTraining data distribution:')
print(train_data['label'].value_counts())
print('\nFirst few training examples:')
print(train_data.head())

# Data preprocessing and model building
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Prepare the data
X_train = train_data['message']
y_train = train_data['label']
X_test = test_data['message']
y_test = test_data['label']

# Encode labels (ham=0, spam=1)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print('Label encoding:', dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Tokenize and pad sequences
max_words = 10000
max_length = 150

tokenizer = Tokenizer(num_words=max_words, oov_token='')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, truncating='post')

print(f'Training sequences shape: {X_train_pad.shape}')
print(f'Test sequences shape: {X_test_pad.shape}')

# Build the model
model = Sequential([
    Embedding(max_words, 128, input_length=max_length),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X_train_pad, y_train_encoded,
    batch_size=32,
    epochs=10,
    validation_data=(X_test_pad, y_test_encoded),
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_encoded, verbose=0)
print(f'\nTest accuracy: {test_accuracy:.4f}')

# Function to predict messages based on model
# (returns list containing prediction probability and label, e.g. [0.0083, 'ham'])
def predict_message(pred_text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded = pad_sequences(sequence, maxlen=max_length, truncating='post')

    # Get prediction probability
    pred_probability = model.predict(padded, verbose=0)[0][0]

    # Determine label based on probability
    # If probability > 0.5, it's spam (1); otherwise, it's ham (0)
    if pred_probability > 0.5:
        pred_label = 'spam'
    else:
        pred_label = 'ham'

    return [float(pred_probability), pred_label]

# Example prediction
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)

# Test function for model and prediction
def test_predictions():
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too"
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()
