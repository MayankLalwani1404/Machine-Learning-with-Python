# Machine-Learning-with-Python

##This repository contains all the projects required to get the Machine Learning with Python certification.

# Project 1: Rock Paper Scissors AI Bot

An advanced AI bot that defeats four different opponent strategies with a 60%+ win rate against each. Built for the FreeCodeCamp Machine Learning with Python certification.

## 🎯 Challenge Overview
Create an AI that can consistently beat four distinct Rock Paper Scissors opponents, each with unique behavioral patterns and strategies.

## 🤖 Opponent Analysis & Strategies

### 1. Quincy - Pattern Follower
**Behavior**: Fixed repeating sequence (R, R, P, P, S)
**Counter Strategy**: Track position in sequence and predict next move
**Win Rate**: 80-99%

### 2. Kris - Reactive Player  
**Behavior**: Always counters player's previous move
**Counter Strategy**: Counter-counter approach - predict what Kris thinks we'll play
**Win Rate**: 60-80%

### 3. Mrugesh - Frequency Analyzer
**Behavior**: Analyzes player's move frequency, counters most common move
**Counter Strategy**: Track our own patterns, predict Mrugesh's counter, then counter that
**Win Rate**: 60-75%

### 4. Abbey - Markov Chain Strategist
**Behavior**: Sophisticated 2-move pattern analysis using Markov chains
**Counter Strategy**: Simulate Abbey's algorithm to predict her prediction, then counter
**Win Rate**: 60-80%

## 🧠 Key Implementation Features

### Adaptive Opponent Recognition
- Behavioral pattern matching to classify opponent types
- Dynamic strategy switching based on opponent detection
- Robust fallback mechanisms for uncertain classifications

### State Management
- Proper handling of game state between matches
- Mutable default parameters for persistent memory
- Clean reset capabilities for new opponents

### Pattern Analysis Algorithms
- **Frequency Analysis**: Track and predict move distributions
- **Sequence Detection**: Identify repeating patterns
- **Markov Chain Simulation**: Replicate complex prediction algorithms
- **Counter-Strategy Logic**: Multi-level prediction and counter-prediction

## 🔧 Technical Details

### Core Algorithm
```python
def player(prev_play, opponent_history=[]):
    # Opponent detection and pattern analysis
    # Strategy selection based on opponent type
    # Move prediction and counter-move generation
    return optimal_move
```

### Success Metrics
- **Minimum Requirement**: 60% win rate against each opponent
- **Achieved Performance**: 60-99% win rates across all opponents
- **Total Matches**: 1000+ games per opponent for validation

## 🏆 Results
✅ **Quincy**: 80-99% win rate (pattern exploitation)  
✅ **Kris**: 60-80% win rate (counter-counter strategy)  
✅ **Mrugesh**: 60-75% win rate (frequency manipulation)  
✅ **Abbey**: 60-80% win rate (Markov chain simulation)  

## 🛠️ Technologies Used
- **Python** - Core implementation
- **Game Theory** - Strategic analysis
- **Pattern Recognition** - Behavioral detection
- **Statistical Analysis** - Move frequency tracking

## 💡 Key Insights
- Different opponents require completely different approaches
- Pattern recognition is crucial for opponent classification
- Multi-level thinking (predicting the opponent's prediction) is essential
- State persistence between games is critical for learning opponent behavior

Perfect for understanding game theory, pattern recognition, and adaptive AI strategies in competitive environments.

# Project 2: Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of cats and dogs. Achieved 68% accuracy, surpassing the 63% requirement and approaching the 70% extra credit goal. Built for the FreeCodeCamp Machine Learning with Python certification.

## 🎯 Challenge Overview
Build a CNN that can accurately classify images as either cats or dogs with at least 63% accuracy on the test set.

## 📊 Dataset Specifications
- **Training Set**: 2,000 images (1,000 cats + 1,000 dogs)
- **Validation Set**: 1,000 images (500 cats + 500 dogs)  
- **Test Set**: 50 unlabeled images for final evaluation
- **Image Format**: 150×150 pixels, RGB color channels
- **Source**: Subset of the classic Kaggle Dogs vs. Cats dataset

## 🏗️ Model Architecture

### Deep CNN with VGG-Style Blocks
```
Input (150×150×3)
    ↓
Conv Block 1: Conv2D(32) → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: Conv2D(128) → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dense(1, sigmoid)
```

### Model Parameters
- **Total Parameters**: ~4.44 million trainable parameters
- **Convolutional Layers**: 6 layers with increasing filter sizes (32→64→128)
- **Fully Connected**: 3 dense layers with dropout regularization
- **Output**: Single sigmoid neuron for binary classification

## ⚙️ Key Implementation Features

### Data Augmentation Pipeline
```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### Regularization Strategies
- **Dropout Layers**: 25% after conv blocks, 50% in dense layers
- **Data Augmentation**: Prevents overfitting on limited dataset
- **Early Stopping**: Monitor validation loss to prevent overtraining

### Training Configuration
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Binary crossentropy
- **Batch Size**: 128 samples
- **Epochs**: 25 (with early stopping capability)
- **Metrics**: Accuracy tracking throughout training

## 📈 Performance Results

### Final Metrics
- **Test Accuracy**: 68% ✅
- **Validation Accuracy**: ~65-70% during training
- **Status**: **PASSED** (Required: 63%, Extra Credit: 70%)
- **Training Time**: ~25 epochs, converging around epoch 20

### Learning Curves
- Steady improvement in validation accuracy
- Effective overfitting prevention through regularization
- Stable convergence without significant loss spikes

## 🔧 Technical Highlights

### CNN Architecture Benefits
- **Feature Extraction**: Hierarchical learning from edges to complex patterns
- **Translation Invariance**: Robust to object position in images
- **Parameter Sharing**: Efficient learning with convolutional filters

### Data Processing Pipeline
- **Normalization**: Pixel values scaled to [0,1] range
- **Real-time Augmentation**: On-the-fly image transformations
- **Memory Efficiency**: Batch processing to handle large datasets

## 🛠️ Technologies Used
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API
- **Python** - Core programming language
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization for training analysis
- **ImageDataGenerator** - Data augmentation and preprocessing

## 💡 Key Insights
- **Data augmentation is crucial** for small datasets to prevent overfitting
- **VGG-style architecture** with double conv layers improves feature extraction
- **Strategic dropout placement** balances regularization and learning capacity
- **Batch normalization** could further improve convergence and accuracy

## 🚀 Potential Improvements
- Implement transfer learning with pre-trained models (VGG16, ResNet)
- Add batch normalization layers for faster convergence
- Experiment with different optimizers (SGD with momentum)
- Increase dataset size or use more aggressive augmentation

Perfect for understanding computer vision, CNN architectures, and image classification fundamentals.

# Project 3: Book Recommendation System (K-Nearest Neighbors)

A collaborative filtering book recommendation system using K-Nearest Neighbors on the Book-Crossings dataset. Suggests similar books based on user ratings using cosine distance similarity. Built for the FreeCodeCamp Machine Learning with Python certification.

## 🎯 Challenge Overview
Develop a book recommendation function that returns 5 similar books to any given book title, based on user rating patterns and collaborative filtering techniques.

## 📚 Dataset: Book-Crossings
- **Source**: Book-Crossings dataset with user ratings
- **Original Size**: ~1.1 million ratings from ~278K users on ~271K books
- **Filtered Dataset**: Users with ≥200 ratings, Books with ≥100 ratings
- **Final Matrix**: Sparse title-by-user rating matrix
- **Encoding**: ISO-8859-1 with semicolon delimiters

## 🔧 Data Preprocessing Challenges

### CSV Parsing Issues
- **Problem**: Single-column parsing due to delimiter/encoding conflicts
- **Solution**: Custom cleaning script to handle semicolons, commas, and special characters
- **Encoding Fix**: Proper handling of ISO-8859-1 character set

### Data Filtering Strategy
```python
# Filter criteria
users_with_200_plus = ratings_per_user >= 200
books_with_100_plus = ratings_per_book >= 100

# Create filtered dataset
filtered_data = original_data[user_filter & book_filter]
```

## 🏗️ Recommendation Algorithm

### Matrix Construction
- **Structure**: Title-by-User rating matrix
- **Values**: Mean ratings per user for each book
- **Missing Data**: Filled with 0 (preserves zero ratings post-filtering)
- **Alternative Considered**: Binary co-rating matrix (discarded for rating-based approach)

### K-Nearest Neighbors Implementation
```python
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(
    n_neighbors=6,  # Include self + 5 recommendations
    metric='cosine',
    algorithm='brute'
)
```

### Distance Metric
- **Metric**: Cosine distance (not similarity)
- **Advantage**: Handles sparse matrices effectively
- **Output**: Raw cosine distances for ranking

## 🎯 Core Function: `get_recommends(title)`

### Function Behavior
```python
def get_recommends(book_title):
    """
    Returns: [book_title, [list_of_5_similar_books_with_distances]]
    Format: ['Book Title', [['Similar Book 1', distance1], ...]]
    """
```

### Implementation Details
- **Input Validation**: Handle non-existent book titles
- **Self-Exclusion**: Remove the input book from recommendations
- **Ranking**: Sort by cosine distance (ascending = more similar)
- **Output Format**: Reversed list order (grader requirement)

## 🔧 Technical Challenges & Solutions

### Edge Case Handling
- **Missing Books**: Graceful handling of books not in the filtered dataset
- **Empty Matrices**: Protection against over-filtering
- **Self-Recommendation**: Automatic exclusion of input book

### Grader Compatibility
- **Distance Format**: Raw cosine distances (not converted to similarity)
- **List Ordering**: Reversed recommendation list for expected output format
- **Exact Format**: Precise list structure matching test requirements

## 📈 Algorithm Performance

### Similarity Detection
- **Effectiveness**: Successfully identifies books with similar rating patterns
- **Scalability**: Efficient with sparse matrices using cosine distance
- **Accuracy**: Verified against expected test cases

### Memory Efficiency
- **Sparse Handling**: Optimized for datasets with many missing ratings
- **Filtering Impact**: Reduced matrix size while maintaining recommendation quality

## 🛠️ Technologies Used
- **scikit-learn** - NearestNeighbors algorithm and cosine distance
- **Pandas** - Data manipulation and CSV handling
- **NumPy** - Matrix operations and numerical computing
- **Python** - Core implementation language

## 💡 Key Technical Insights

### Collaborative Filtering Principles
- **User Similarity**: Books rated similarly by the same users are related
- **Cosine Distance**: Effective for high-dimensional sparse rating vectors
- **Rating Patterns**: More informative than simple binary co-occurrence

### Data Quality Impact
- **Filtering Threshold**: Balance between data quality and matrix density
- **Rating Distribution**: Mean ratings provide better similarity than binary indicators
- **User Activity**: High-activity users contribute more reliable similarity signals

## 🔍 Verification & Testing
- **Test Cases**: Verified with expected top 5 recommendations
- **Format Compliance**: Exact match with grader requirements
- **Edge Cases**: Robust handling of invalid inputs and missing data

## 🚀 Potential Enhancements
- **Matrix Factorization**: SVD or NMF for dimensionality reduction
- **Hybrid Approaches**: Combine content-based and collaborative filtering
- **Deep Learning**: Neural collaborative filtering for non-linear patterns
- **Real-time Updates**: Incremental learning for new ratings

Perfect for understanding collaborative filtering, similarity metrics, and recommendation system fundamentals in machine learning.

# Project 4: Healthcare Cost Prediction with Neural Networks

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

# Project 5: SMS Spam Classification with LSTM

A deep learning model that classifies SMS messages as legitimate ("ham") or spam using TensorFlow/Keras. Built for the FreeCodecamp Machine Learning certification challenge.

## 🚀 Features
- **LSTM Neural Network** with 1.3M parameters for sequential text processing
- **97-99% accuracy** on SMS Spam Collection dataset
- **Real-time prediction** with confidence scores
- **Robust preprocessing** with OOV handling and sequence padding

## 📊 Dataset
- **5,574 SMS messages** (87% ham, 13% spam)
- Text sequences processed to 150 tokens max
- 10,000-word vocabulary with embedding layer

## 🏗️ Architecture
- Embedding (10k vocab → 128d) → LSTM (64 units) → Dense (32) → Sigmoid (1)

## 🔧 Tech Stack
- **TensorFlow/Keras** - Deep learning framework
- **Python** - Core language
- **scikit-learn** - Label encoding
- **Pandas/NumPy** - Data manipulation

## 📈 Performance
- **Binary Classification** with 0.5 threshold
- **Adam optimizer** with binary crossentropy loss
- **Dropout regularization** (50%) to prevent overfitting

The model achieves high accuracy due to the LSTM's ability to capture sequential patterns in text that distinguish spam (promotional language, urgency, contact info) from legitimate personal communications.

## 💡 Usage
```python
prediction = predict_message("Your SMS text here")
# Returns: [probability_score, "ham"/"spam"]
