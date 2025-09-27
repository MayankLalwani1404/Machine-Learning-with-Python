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
