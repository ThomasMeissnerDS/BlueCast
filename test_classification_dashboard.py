#!/usr/bin/env python3
"""
Simple test for the classification dashboard.
"""

from bluecast.eda.analyse import create_eda_dashboard_classification
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


def main():
    # Create sample data
    df = create_synthetic_dataframe(num_samples=500, random_state=42)
    print("Dataset created with columns:", list(df.columns))
    print("Target classes:", df["target"].unique())
    print("Shape:", df.shape)

    # Start classification dashboard
    print("Starting Professional Classification Dashboard on http://localhost:8051")
    print("")
    print("🎨 NEW FEATURES:")
    print("✨ Beautiful dark theme with professional styling")
    print("🎯 Target Distribution: Shows class balance with emojis")
    print("🎨 Scatter by Class: X vs Y features colored by target class")
    print("📦 Box Plot by Class: Feature distribution across classes")
    print("📊 Feature by Target: Shows how features vary by target class")
    print("🔗 Enhanced correlation heatmaps with dark theme")
    print("🎯 PCA analysis with color-coded classes")
    print("")
    print("🚀 IMPROVEMENTS:")
    print("- Dual feature selection (X and Y independently)")
    print("- Professional gradient headers")
    print("- Responsive design with better spacing")
    print("- Enhanced tooltips and navigation")
    print("- Beautiful color palettes")
    print("")
    print("📋 Instructions:")
    print("1. Select plot type from the dropdown")
    print("2. Select Feature X and Feature Y independently")
    print("3. Explore the beautiful visualizations!")
    print("")
    print("Press Ctrl+C to stop and see the magic! ✨")

    create_eda_dashboard_classification(
        df=df, target_col="target", port=8051, run_server=True
    )


if __name__ == "__main__":
    main()
