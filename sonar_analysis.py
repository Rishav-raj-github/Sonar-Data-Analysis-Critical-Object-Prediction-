#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sonar Data Analysis for Critical Object Prediction.

This module analyzes sonar data to differentiate between rocks and mines
using supervised machine learning (Logistic Regression).

Author: Rishav Raj
Date: 2023
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_sonar_data(filepath, header=None):
    """
    Load sonar dataset from CSV file.

    Args:
        filepath (str): Path to the CSV file containing sonar data
        header (int, optional): Row number to use as column names. Defaults to None.

    Returns:
        pd.DataFrame: Loaded sonar dataset
    """
    try:
        data = pd.read_csv(filepath, header=header)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None


def explore_data(data, target_column=60):
    """
    Perform exploratory data analysis on sonar dataset.

    Args:
        data (pd.DataFrame): Sonar dataset
        target_column (int): Column index for target variable

    Returns:
        None
    """
    print("\n=== Dataset Shape ===")
    print(f"Shape: {data.shape}")

    print("\n=== First Few Rows ===")
    print(data.head())

    print("\n=== Statistical Summary ===")
    print(data.describe())

    print("\n=== Target Variable Distribution ===")
    print(data[target_column].value_counts())

    print("\n=== Mean Values Grouped by Target ===")
    print(data.groupby(target_column).mean())


def prepare_data(data, target_column=60, test_size=0.1, random_state=1):
    """
    Prepare data for training by splitting features and target.

    Args:
        data (pd.DataFrame): Sonar dataset
        target_column (int): Column index for target variable
        test_size (float): Proportion of dataset for test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = data.drop(columns=target_column, axis=1)
    y = data[target_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train logistic regression model on training data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target labels

    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on training and test data.

    Args:
        model: Trained machine learning model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target labels

    Returns:
        tuple: Training accuracy, test accuracy
    """
    # Predict on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on test data
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n=== Model Evaluation ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return train_accuracy, test_accuracy


def predict_object(model, input_data):
    """
    Predict whether input sonar data represents a rock or mine.

    Args:
        model: Trained machine learning model
        input_data (tuple or list): Sonar reading features (60 values)

    Returns:
        str: Prediction result ('Rock' or 'Mine')
    """
    # Convert input to numpy array and reshape
    input_array = np.asarray(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    # Interpret result
    if prediction[0] == 'R':
        result = 'Rock'
    else:
        result = 'Mine'

    print(f"\n=== Prediction ===")
    print(f"Prediction: {prediction[0]}")
    print(f"The object is a {result}")

    return result


def main():
    """
    Main function to execute the complete sonar data analysis pipeline.
    """
    print("=" * 60)
    print("Sonar Data Analysis - Rock vs Mine Classification")
    print("=" * 60)

    # Load data
    filepath = 'Copy of sonar data.csv'
    data = load_sonar_data(filepath, header=None)

    if data is None:
        print("\nExiting due to data loading error.")
        return

    # Explore data
    explore_data(data)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Example prediction
    sample_input = (
        0.0124, 0.0433, 0.0604, 0.0449, 0.0597, 0.0355, 0.0531, 0.0343,
        0.1052, 0.2120, 0.1640, 0.1901, 0.3026, 0.2019, 0.0592, 0.2390,
        0.3657, 0.3809, 0.5929, 0.6299, 0.5801, 0.4574, 0.4449, 0.3691,
        0.6446, 0.8940, 0.8978, 0.4980, 0.3333, 0.2350, 0.1553, 0.3666,
        0.4340, 0.3082, 0.3024, 0.4109, 0.5501, 0.4129, 0.5499, 0.5018,
        0.3132, 0.2802, 0.2351, 0.2298, 0.1155, 0.0724, 0.0621, 0.0318,
        0.0450, 0.0167, 0.0078, 0.0083, 0.0057, 0.0174, 0.0188, 0.0054,
        0.0114, 0.0196, 0.0147, 0.0062
    )

    predict_object(model, sample_input)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
