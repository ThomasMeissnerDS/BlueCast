#!/usr/bin/env python3
"""
Simple test for the regression dashboard.
"""

from bluecast.tests.make_data.create_data import create_synthetic_dataframe_regression
from bluecast.eda.analyse import create_eda_dashboard_regression

def main():
    # Create sample data
    df = create_synthetic_dataframe_regression(num_samples=500, random_state=42)
    print('Regression Dataset created with columns:', list(df.columns))
    print('Target range: {:.2f} to {:.2f}'.format(df['target'].min(), df['target'].max()))
    print('Shape:', df.shape)
    
    # Start regression dashboard
    print('Starting Regression Dashboard on http://localhost:8052')
    print('Features available in this dashboard:')
    print('- Correlation Heatmap: Feature correlations')
    print('- Scatter with Regression: X vs Y with regression line and train/test split')
    print('- Feature Coefficients: Linear regression coefficients visualization')
    print('- Distribution Plot: Feature distributions')
    print('- Plus standard EDA plots (PCA, Box plots, etc.)')
    print('')
    print('Instructions:')
    print('1. Select plot type from dropdown')
    print('2. For "Scatter with Regression", select Feature X and Feature Y')
    print('3. Try "Feature Coefficients" to see which features are most important')
    print('4. The regression line shows train (blue) vs test (red) data split')
    print('')
    print('Press Ctrl+C to stop')
    
    app = create_eda_dashboard_regression(
        df=df, 
        target_col='target', 
        port=8052, 
        run_server=True
    )

if __name__ == "__main__":
    main() 