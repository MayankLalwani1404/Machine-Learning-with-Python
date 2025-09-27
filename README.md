# Machine-Learning-with-Python

This repository contains all the projects required to get the Machine Learning with Python certification.

Project 1: Rock, Paper, Scissors AI bot

Description:
This project implements an advanced Rock Paper Scissors AI that can defeat four different opponent strategies with a win rate of at least 60% against each.

Solution Strategy

Opponent Detection System
The bot uses pattern analysis to identify which of four opponent types it's facing:

1. Quincy: Follows a fixed repeating pattern (R, R, P, P, S)
2. Kris: Always plays the counter to the player's previous move
3. Mrugesh: Analyzes the player's move frequency and counters the most common move
4. Abbey: Uses sophisticated Markov chain analysis of 2-move patterns to predict the next move

Counter Strategies

•  vs Quincy: Predicts the next move in the fixed sequence and plays the counter
•  vs Kris: Plays what beats Kris's counter to our previous move (counter-counter strategy)
•  vs Mrugesh: Tracks our own move frequency and counters Mrugesh's predicted counter
•  vs Abbey: Simulates Abbey's pattern analysis algorithm to predict what Abbey thinks we'll play, then counters Abbey's counter move

Key Implementation Features

•  Adaptive Opponent Recognition: Uses behavioral pattern matching to classify opponents
•  State Management: Properly handles game state between matches using mutable default parameters
•  Fallback Strategies: Robust handling when opponent classification is uncertain
•  Pattern Analysis: Implements frequency analysis and Markov chain simulation

Results
The AI successfully achieves 60%+ win rates against all four opponent types, with some strategies achieving 80-99% win rates.

Project 2: Cat vs Dog Image Classifier

Overview
A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of cats and dogs. Achieved 68% accuracy, surpassing the 63% requirement and approaching the 70% extra credit goal.

Dataset
•  Training: 2,000 images (1,000 cats + 1,000 dogs)
•  Validation: 1,000 images (500 cats + 500 dogs)  
•  Test: 50 unlabeled images
•  Image size: 150×150 pixels, RGB format

Model Architecture
Deep CNN with VGG-style blocks:
•  Conv Block 1: 32→32 filters, MaxPool, Dropout(0.25)
•  Conv Block 2: 64→64 filters, MaxPool, Dropout(0.25)
•  Conv Block 3: 128→128 filters, MaxPool, Dropout(0.25)
•  Dense Layers: 512→256→1 with Dropout(0.5)
•  Output: Sigmoid activation for binary classification
•  Total Parameters: ~5.8 million

Key Features
✅ Data Augmentation: Rotation, shifts, shear, zoom, horizontal flip  
✅ Regularization: Strategic dropout placement to prevent overfitting  
✅ Optimized Training: Adam optimizer with 0.0001 learning rate, 25 epochs  
✅ VGG Architecture: Double conv layers for better feature extraction  

Results
•  Final Accuracy: 68% on test set
•  Status: ✅ PASSED (Required: 63%, Extra Credit: 70%)
•  Training Time: ~25 epochs with batch size 128

Technologies
•  TensorFlow 2.x & Keras
•  Python with NumPy, Matplotlib
•  Image Processing with ImageDataGenerator

Project 3: Book Recommendation System (K-Nearest Neighbors)

This project develops a book recommendation system using K-Nearest Neighbors (KNN) on the Book-Crossings dataset. The goal is to suggest similar books based on user ratings, utilizing a title-by-user matrix and cosine distance to find the nearest neighbors. Key highlights include:

Data Cleaning & Preprocessing:

Tackled CSV parsing issues (single-column problem) caused by delimiter/encoding quirks.

Developed a cleaning script to standardize the data, ensuring proper handling of semicolons, commas, and ISO-8859-1 encoding.

Filtered the data to include only users with ≥200 ratings and books with ≥100 ratings.

Matrix Construction:

Built a title-by-user rating matrix using mean ratings, filling missing values with 0 to preserve zero ratings post-filtering.

Evaluated both binary co-rating and average rating approaches before settling on the latter.

K-Nearest Neighbors Model:

Implemented KNN with a cosine distance metric using the NearestNeighbors algorithm.

Ensured results were returned as raw cosine distances (not similarity) and reversed the recommendation list to match grader expectations.

API for Book Recommendations:

Developed a function get_recommends(title) that returns 5 similar books to a given title, along with their cosine distances.

Managed edge cases such as self-recommendation and empty matrices from over-filtering.

Verification:

Successfully verified the model's output with expected top 5 recommendations for test cases, adhering to challenge constraints (including reversed order for grading purposes).

# Healthcare Cost Prediction with Neural Networks

A machine learning project that predicts individual healthcare insurance costs using demographic and health factors through deep neural network regression.

## 📊 Project Overview

This project implements a neural network to predict healthcare costs based on personal attributes. The model successfully achieves a Mean Absolute Error of **$2,879.97**, meeting the challenge requirement of MAE < $3,500.

## 🎯 Key Results

- **Model Accuracy**: 78.3% prediction accuracy
- **Mean Absolute Error**: $2,879.97
- **Dataset Size**: 1,338 insurance records
- **Features**: 9 processed features from demographic and health data
- **Neural Network**: 4-layer deep learning model with 11,649 parameters

## 📁 Dataset Features

The model uses the following input features:
- **Demographics**: Age (18-64), Sex, Region (4 US regions)
- **Health Metrics**: BMI (16.0-53.1), Number of children (0-5)
- **Lifestyle**: Smoking status (primary cost predictor)
- **Target**: Healthcare insurance expenses ($1,122 - $63,770)

## 🧠 Model Architecture

```
Input Layer:    9 → 128 neurons (ReLU + 30% Dropout)
Hidden Layer 1: 128 → 64 neurons (ReLU + 30% Dropout)  
Hidden Layer 2: 64 → 32 neurons (ReLU)
Output Layer:   32 → 1 neuron (Linear regression)
Total Parameters: 11,649
```

## 🚀 Technical Implementation

- **Framework**: TensorFlow/Keras
- **Preprocessing**: StandardScaler normalization, one-hot encoding
- **Train/Test Split**: 80/20 (1,070/268 samples)
- **Optimization**: Adam optimizer with MSE loss
- **Training**: 200 epochs with validation monitoring
- **Regularization**: Dropout layers to prevent overfitting

## 📈 Performance Metrics

- **Mean Absolute Error**: $2,879.97
- **Error Rate**: 21.7% of mean healthcare cost
- **Range Coverage**: 4.6% of total cost range
- **Feature Correlation**: Smoking (0.787), Age (0.299), BMI (0.199)

## 💻 Usage

```python
# Load and preprocess data
dataset = pd.read_csv('insurance.csv')
# ... preprocessing steps

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[9]),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Train and evaluate
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_dataset, train_labels, epochs=200)
```

## 🏆 Challenge Completion

✅ **Passed FreeCodeCamp Machine Learning Challenge**
- Required: MAE < $3,500
- Achieved: MAE = $2,879.97
- Success margin: $620.03 below requirement

## 📋 Files

- `fcc_predict_health_costs_with_regression.ipynb` - Main Jupyter notebook
- `insurance.csv` - Healthcare cost dataset
- `README.md` - Project documentation

## 🛠️ Dependencies

```
tensorflow>=2.0
pandas
numpy
matplotlib
scikit-learn
```

## 📊 Key Insights

- **Smoking status** is the strongest predictor of healthcare costs (5x impact)
- **Age and BMI** show moderate correlation with expenses  
- **Geographic region** has minimal impact on costs
- **Neural network** effectively captures non-linear relationships

---

*This project demonstrates proficiency in deep learning, data preprocessing, and regression analysis for healthcare cost prediction.*
