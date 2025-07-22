#!/usr/bin/env python3
"""
Simple test for the regression dashboard.
"""

from bluecast.eda.analyse import create_eda_dashboard_regression
from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression


def main():
    # Create sample data
    df = create_synthetic_dataframe_regression(num_samples=500, random_state=42)
    print("Regression Dataset created with columns:", list(df.columns))
    print(
        "Target range: {:.2f} to {:.2f}".format(df["target"].min(), df["target"].max())
    )
    print("Shape:", df.shape)

    # Start regression dashboard
    print("Starting Professional Regression Dashboard on http://localhost:8052")
    print("")
    print("🔬 NEW FEATURES:")
    print("✨ Sleek dark theme with gradient headers")
    print("📊 Scatter with Regression: Train/test split with R² scores")
    print("⚖️ Feature Coefficients: Beautiful bar charts showing importance")
    print("🔗 Enhanced correlation heatmaps")
    print("📈 Professional distribution plots")
    print("🎯 PCA analysis with better styling")
    print("")
    print("🚀 IMPROVEMENTS:")
    print("- 🔵 Blue for training data")
    print("- 🔴 Red for test data")
    print("- 📈 Green regression lines")
    print("- Professional color schemes")
    print("- Responsive layout design")
    print("- Enhanced dataset information display")
    print("")
    print("📋 Instructions:")
    print("1. Select plot type from the dropdown")
    print('2. For "Scatter with Regression", select Feature X and Feature Y')
    print('3. Try "Feature Coefficients" to see feature importance')
    print("4. Watch the beautiful regression lines and R² scores!")
    print("")
    print("Press Ctrl+C to stop and admire the results! 🚀")

    create_eda_dashboard_regression(
        df=df, target_col="target", port=8052, run_server=True
    )


if __name__ == "__main__":
    main()
