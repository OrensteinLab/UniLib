# DNA Sequence Analysis Pipeline

This repository contains a Python script for analyzing DNA sequences using a Transfer Learning technique. The script reads files with expression data of many regulatory DNA sequences, preprocesses the data, trains a convolutional neural network (CNN) model with a transfer learning technique, and performs predictions on both test and validation sets. 

## Prerequisites

Before using the script, make sure you have the following libraries and tools installed:

- [Python](https://www.python.org/): The script is written in Python and requires a Python interpreter.
- [Pandas](https://pandas.pydata.org/): This library is used for data manipulation and handling data frames.
- [NumPy](https://numpy.org/): NumPy is used for numerical operations, including matrix and array operations.
- [TensorFlow](https://www.tensorflow.org/): TensorFlow is used for building and training machine learning models.
- [SciPy](https://www.scipy.org/): SciPy is used for scientific and technical computing and includes statistical functions.

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
   - Implements an ensemble method for model prediction. The script runs 100 iterations of model prediction to improve performance.

5. **Prediction**:
   - Uses the trained or pretrained model to predict mean FL values for test and validation sequences.
   - Calculates predictions based on  the average predictions of the original and reverse complement sequences, improving prediction accuracy.

6. **Evaluation**:
   - Calculates Pearson correlation coefficients between the model predictions and the true labels for the 2 different validation set.
   - Evaluates model performance.

7. **Data Export**:
   - Saves the average predictions and true leabels in CSV files for further analysis.

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

5. **Review Results**:
   - Review the Pearson correlation coefficients to assess the model's performance.
   - Examine the CSV files with predictions results for further analysis.
