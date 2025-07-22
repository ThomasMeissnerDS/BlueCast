#!/usr/bin/env python3
"""
Simple test for the classification dashboard.
"""

from bluecast.tests.make_data.create_data import create_synthetic_dataframe
from bluecast.eda.analyse import create_eda_dashboard_classification

def main():
    # Create sample data
    df = create_synthetic_dataframe(num_samples=500, random_state=42)
    print('Dataset created with columns:', list(df.columns))
    print('Target classes:', df['target'].unique())
    print('Shape:', df.shape)
    
    # Start classification dashboard
    print('Starting Classification Dashboard on http://localhost:8051')
    print('Features available in this dashboard:')
    print('- Target Distribution: Shows class balance')
    print('- Scatter by Class: X vs Y features colored by target class')
    print('- Box Plot by Class: Feature distribution across classes')
    print('- Feature by Target: Shows how features vary by target class')
    print('- Plus standard EDA plots (correlation, PCA, etc.)')
    print('')
    print('Instructions:')
    print('1. Select plot type from dropdown')
    print('2. Select Feature X and Feature Y for scatter plots')
    print('3. Explore different combinations!')
    print('')
    print('Press Ctrl+C to stop')
    
    app = create_eda_dashboard_classification(
        df=df, 
        target_col='target', 
        port=8051, 
        run_server=True
    )

if __name__ == "__main__":
    main() 