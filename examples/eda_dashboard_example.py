"""
Example script demonstrating the EDA dashboard functionality in BlueCast.

This script shows how to:
1. Create sample data
2. Use individual plotting functions with plotly
3. Create an interactive dashboard with just a few lines of code

Run this script with: python examples/eda_dashboard_example.py
"""

import numpy as np
import pandas as pd
from bluecast.eda.analyse import (
    correlation_heatmap,
    plot_pca,
    plot_pie_chart,
    mutual_info_to_target,
    create_eda_dashboard,
)


def create_sample_data(n_samples=1000):
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create features
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    }
    
    # Create target based on features (binary classification)
    target_prob = (
        0.1 * (data['age'] - 50) / 30 +
        0.2 * np.log(data['income']) / 10 +
        0.15 * (data['education_years'] - 12) / 8 +
        0.1 * (data['experience'] - 20) / 20 +
        np.random.normal(0, 0.3, n_samples)
    )
    data['target'] = (target_prob > np.median(target_prob)).astype(int)
    
    return pd.DataFrame(data)


def demonstrate_individual_plots(df):
    """Demonstrate individual plotting functions."""
    print("Demonstrating individual plotting functions...")
    
    # 1. Correlation heatmap
    print("Creating correlation heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig = correlation_heatmap(df[numeric_cols], show=False)
    fig.show()
    
    # 2. PCA plot
    print("Creating PCA plot...")
    fig = plot_pca(df[list(numeric_cols)], 'target', show=False)
    fig.show()
    
    # 3. Pie chart for categorical data
    print("Creating pie chart...")
    fig = plot_pie_chart(df, 'category', show=False)
    fig.show()
    
    # 4. Mutual information plot
    print("Creating mutual information plot...")
    fig = mutual_info_to_target(
        df.select_dtypes(include=[np.number]), 
        'target', 
        'binary',
        show=False
    )
    fig.show()
    
    print("Individual plots demonstration complete!")


def main():
    """Main function demonstrating the EDA capabilities."""
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data(1000)
    print(f"Created dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Demonstrate individual plotting functions
    demonstrate_individual_plots(df)
    
    # Create and launch the dashboard
    print("\nCreating EDA dashboard...")
    print("This will open a web browser with an interactive dashboard.")
    print("Navigate to http://localhost:8050 to view the dashboard.")
    print("Press Ctrl+C to stop the dashboard server.")
    
    try:
        # Note: run_server=True is the default, explicitly shown here for clarity
        create_eda_dashboard(df, 'target', port=8050, run_server=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        print("Make sure you have dash installed: pip install dash")


if __name__ == "__main__":
    main() 