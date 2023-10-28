# DNA Sequence Analysis Pipeline

This repository contains a Python script for analyzing DNA sequences using the Transfer Learning Technique. The script reads files with expression data of many regulatory DNA sequences, preprocesses the data, trains a convolutional neural network (CNN) model with the transfer learning technique, and performs predictions on 2 validation sets with regulatory DNA sequences.

## Prerequisites

Before using the script, make sure you have the following libraries and tools installed:

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [SciPy](https://www.scipy.org/)
## Script Description

The script performs the following main tasks:

1. **Data Preprocessing**:
   - Reads a CSV file containing DNA sequences and expression measurements (mean fluorescence levels).
   - Converts the DNA sequences to one-hot encodings, enabling them to be used as input to the machine learning model.
   - Prepares reverse complement sequences for training and prediction.

2. **Machine Learning Model**:
   - Defines a convolutional neural network (CNN) model for the task.
   - Trains model with transfer learning technique on 3 datasets with different sizes and data quality
   - Compiles the model with appropriate loss and optimization functions.

3. **Model Training**:
   - Trains the CNN model on the preprocessed DNA sequences and their corresponding mean FL levels.
   - Supports training on different datasets.

4. **Ensemble Model**:
   - Implements an ensemble method for model prediction. The script runs 100 models with different initializations and averages their predictions to improve performance.

5. **Prediction**:
   - Uses the trained model to predict mean FL values validation sequences.
   - Calculates predictions based on the average predictions for the original and reverse complement sequences, improving prediction accuracy.

6. **Evaluation**:
   - Calculates Pearson correlation coefficients between the model predictions and the true labels for the 2 different validation set.
   - Evaluates model performance.

7. **Data Export**:
   - Saves the average predictions and true labels to CSV files for further analysis.

## Usage

To use this script, follow these steps:

1. **Install Required Libraries**:
   Ensure that you have installed all the required libraries and tools mentioned in the "Prerequisites" section.

2. **Data Preparation**:
   - Prepare your input data in CSV format, including DNA sequences and associated target values.
   - Name your input CSV files accordingly, or modify the script to specify the correct file paths.

3. **Configure the Script**:
   - Adjust the script to match your dataset and experimental setup. Update the file paths and parameters as needed.

4. **Run the Script**:
   - Execute the script using a Python interpreter.
   - The script will perform data preprocessing, model training, and evaluation.
