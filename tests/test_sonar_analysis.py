#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for sonar_analysis module.

This module contains test cases for the sonar data analysis functions.
Tests use pytest framework.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Mock imports from sonar_analysis since we don't have the actual CSV file
try:
    import sys
    sys.path.insert(0, '..')
    from sonar_analysis import (
        load_sonar_data,
        explore_data,
        prepare_data,
        train_model,
        evaluate_model,
        predict_object
    )
except ImportError:
    # If import fails, define placeholder functions for testing
    pass


class TestLoadSonarData:
    """Test cases for load_sonar_data function."""

    def test_load_data_success(self):
        """Test successful data loading."""
        # Create a mock CSV file data
        mock_data = pd.DataFrame({
            0: [0.02, 0.03],
            1: [0.04, 0.05],
            60: ['R', 'M']
        })

        with patch('pandas.read_csv', return_value=mock_data):
            result = load_sonar_data('test.csv', header=None)
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] == 2

    def test_load_data_file_not_found(self):
        """Test handling of missing file."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            result = load_sonar_data('nonexistent.csv', header=None)
            assert result is None


class TestPrepareData:
    """Test cases for prepare_data function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample sonar data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.rand(100, 61))
        # Set last column as target with binary labels
        data[60] = ['R' if i % 2 == 0 else 'M' for i in range(100)]
        return data

    def test_prepare_data_split(self, sample_data):
        """Test data splitting functionality."""
        X_train, X_test, y_train, y_test = prepare_data(
            sample_data,
            target_column=60,
            test_size=0.2,
            random_state=42
        )

        # Check shapes
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check that features don't contain target column
        assert 60 not in X_train.columns
        assert 60 not in X_test.columns

    def test_prepare_data_stratification(self, sample_data):
        """Test stratified splitting."""
        X_train, X_test, y_train, y_test = prepare_data(
            sample_data,
            target_column=60,
            test_size=0.2,
            random_state=42
        )

        # Check that both classes are present in train and test
        assert 'R' in y_train.values
        assert 'M' in y_train.values
        assert 'R' in y_test.values
        assert 'M' in y_test.values


class TestTrainModel:
    """Test cases for train_model function."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(80, 60))
        y_train = pd.Series(['R' if i % 2 == 0 else 'M' for i in range(80)])
        return X_train, y_train

    def test_train_model_returns_fitted_model(self, sample_training_data):
        """Test that train_model returns a fitted model."""
        X_train, y_train = sample_training_data
        model = train_model(X_train, y_train)

        # Check that model is fitted by attempting prediction
        assert hasattr(model, 'predict')
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)


class TestEvaluateModel:
    """Test cases for evaluate_model function."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model with test data."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(80, 60))
        y_train = pd.Series(['R' if i % 2 == 0 else 'M' for i in range(80)])
        X_test = pd.DataFrame(np.random.rand(20, 60))
        y_test = pd.Series(['R' if i % 2 == 0 else 'M' for i in range(20)])

        model = train_model(X_train, y_train)
        return model, X_train, y_train, X_test, y_test

    def test_evaluate_model_returns_accuracies(self, trained_model_and_data):
        """Test that evaluate_model returns accuracy scores."""
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        train_acc, test_acc = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )

        # Check that accuracies are valid percentages
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1
        assert isinstance(train_acc, float)
        assert isinstance(test_acc, float)


class TestPredictObject:
    """Test cases for predict_object function."""

    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model."""
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(80, 60))
        y_train = pd.Series(['R' if i % 2 == 0 else 'M' for i in range(80)])
        return train_model(X_train, y_train)

    def test_predict_object_with_valid_input(self, trained_model):
        """Test prediction with valid input data."""
        sample_input = tuple(np.random.rand(60))
        result = predict_object(trained_model, sample_input)

        # Check that result is either 'Rock' or 'Mine'
        assert result in ['Rock', 'Mine']

    def test_predict_object_with_array_input(self, trained_model):
        """Test prediction with array input."""
        sample_input = list(np.random.rand(60))
        result = predict_object(trained_model, sample_input)

        assert result in ['Rock', 'Mine']

    def test_predict_object_reshaping(self, trained_model):
        """Test that input is properly reshaped for prediction."""
        sample_input = tuple(np.random.rand(60))

        # This should not raise an error
        result = predict_object(trained_model, sample_input)
        assert result is not None


class TestDataIntegrity:
    """Test cases for data integrity and edge cases."""

    def test_data_dimensions(self):
        """Test that data has correct dimensions."""
        # Create sample data with 60 features + 1 target
        data = pd.DataFrame(np.random.rand(50, 61))
        data[60] = ['R' if i % 2 == 0 else 'M' for i in range(50)]

        X_train, X_test, y_train, y_test = prepare_data(data, target_column=60)

        # Features should have 60 columns
        assert X_train.shape[1] == 60
        assert X_test.shape[1] == 60

    def test_no_data_leakage(self):
        """Test that there's no data leakage between train and test."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.rand(100, 61))
        data[60] = ['R' if i % 2 == 0 else 'M' for i in range(100)]

        X_train, X_test, y_train, y_test = prepare_data(
            data, target_column=60, test_size=0.2, random_state=42
        )

        # Check that train and test sets don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
