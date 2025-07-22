#!/usr/bin/env python3
"""
Enhanced test script for the EDA dashboard functionality.
This script creates synthetic data and launches different dashboard variants for testing.
"""

import os
import sys

from bluecast.eda.analyse import (
    create_eda_dashboard,
    create_eda_dashboard_classification,
    create_eda_dashboard_regression,
)
from bluecast.tests.make_data.create_data import (
    create_synthetic_dataframe,
    create_synthetic_dataframe_regression,
    create_synthetic_multiclass_dataframe,
)

# Add the bluecast directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bluecast"))


def test_classification_dashboard():
    """Test the classification dashboard with binary classification data."""
    print("=" * 60)
    print("TESTING CLASSIFICATION DASHBOARD")
    print("=" * 60)

    # Create binary classification dataset
    df = create_synthetic_dataframe(num_samples=1000, random_state=42)

    print("Binary Classification Dataset created:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("Target column: 'target'")
    print("Target classes:", df["target"].unique())
    print("\nDataset preview:")
    print(df.head())

    print("\n" + "=" * 50)
    print("Starting Classification Dashboard...")
    print("Dashboard will be available at: http://localhost:8051")
    print("Features include:")
    print("- Target Distribution")
    print("- Scatter plots colored by class")
    print("- Box plots by class")
    print("- Feature distribution by target class")
    print("Press Ctrl+C to stop and continue to next test")
    print("=" * 50)

    try:
        create_eda_dashboard_classification(
            df=df, target_col="target", port=8051, run_server=True
        )
    except KeyboardInterrupt:
        print("\nClassification dashboard stopped by user.")


def test_regression_dashboard():
    """Test the regression dashboard with regression data."""
    print("=" * 60)
    print("TESTING REGRESSION DASHBOARD")
    print("=" * 60)

    # Create regression dataset
    df = create_synthetic_dataframe_regression(num_samples=1000, random_state=42)

    print("Regression Dataset created:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("Target column: 'target'")
    print(
        "Target range: {:.2f} to {:.2f}".format(df["target"].min(), df["target"].max())
    )
    print("\nDataset preview:")
    print(df.head())

    print("\n" + "=" * 50)
    print("Starting Regression Dashboard...")
    print("Dashboard will be available at: http://localhost:8052")
    print("Features include:")
    print("- Scatter plots with regression lines")
    print("- Train/test split visualization")
    print("- Feature coefficient visualization")
    print("- R² scores for model performance")
    print("Press Ctrl+C to stop and continue to next test")
    print("=" * 50)

    try:
        create_eda_dashboard_regression(
            df=df, target_col="target", port=8052, run_server=True
        )
    except KeyboardInterrupt:
        print("\nRegression dashboard stopped by user.")


def test_multiclass_dashboard():
    """Test the classification dashboard with multiclass data."""
    print("=" * 60)
    print("TESTING MULTICLASS CLASSIFICATION DASHBOARD")
    print("=" * 60)

    # Create multiclass classification dataset
    df = create_synthetic_multiclass_dataframe(num_samples=1000, random_state=42)

    print("Multiclass Classification Dataset created:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("Target column: 'target'")
    print("Target classes:", sorted(df["target"].unique()))
    print("\nDataset preview:")
    print(df.head())

    print("\n" + "=" * 50)
    print("Starting Multiclass Classification Dashboard...")
    print("Dashboard will be available at: http://localhost:8053")
    print("Features include:")
    print("- Multiple class target distribution")
    print("- Scatter plots with multiple class colors")
    print("- Feature distributions across all classes")
    print("Press Ctrl+C to stop and continue to next test")
    print("=" * 50)

    try:
        create_eda_dashboard_classification(
            df=df, target_col="target", port=8053, run_server=True
        )
    except KeyboardInterrupt:
        print("\nMulticlass dashboard stopped by user.")


def test_original_dashboard():
    """Test the original dashboard for backward compatibility."""
    print("=" * 60)
    print("TESTING ORIGINAL DASHBOARD (Backward Compatibility)")
    print("=" * 60)

    # Create dataset
    df = create_synthetic_dataframe(num_samples=1000, random_state=42)

    print("Original Dashboard Test:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("Target column: 'target'")
    print("\nDataset preview:")
    print(df.head())

    print("\n" + "=" * 50)
    print("Starting Original Dashboard...")
    print("Dashboard will be available at: http://localhost:8054")
    print("This is the original implementation for backward compatibility")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        create_eda_dashboard(df=df, target_col="target", port=8054, run_server=True)
    except KeyboardInterrupt:
        print("\nOriginal dashboard stopped by user.")


def main():
    print("Enhanced EDA Dashboard Test Suite")
    print("This script will test all dashboard variants:")
    print("1. Classification Dashboard (Binary)")
    print("2. Regression Dashboard")
    print("3. Classification Dashboard (Multiclass)")
    print("4. Original Dashboard (Backward Compatibility)")
    print("\nEach dashboard will run on a different port.")
    print("Press Ctrl+C in each dashboard to move to the next test.")
    print("\n" + "=" * 60)

    try:
        # Test all dashboard variants
        test_classification_dashboard()
        test_regression_dashboard()
        test_multiclass_dashboard()
        test_original_dashboard()

        print("\n" + "=" * 60)
        print("All dashboard tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running dashboard tests: {e}")
        print("Make sure you have 'dash' and 'scikit-learn' installed:")
        print("poetry add dash scikit-learn")


if __name__ == "__main__":
    main()
