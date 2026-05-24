import unittest
import pandas as pd
import numpy as np

from feature_engineering import engineer_features
from custom_preprocessor import PerfectFitPreprocessor

class TestPerfectFitPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe matching Kaggle dataset
        self.df_train = pd.DataFrame({
            'x1': [0.1, -0.2, 0.5, 0.0, 0.4],
            'x2': [0.2, 0.1, -0.1, 0.5, 0.3],
            'x4': [0.1, 0.2, 0.3, 0.4, 0.5],
            'x5': [10.0, 999.0, 8.0, 11.0, 999.0],  # 999.0 is sentinel
            'x6': [-5.0, 10.0, 15.0, -10.0, 0.0],
            'x7': [2.0, -2.0, 5.0, 10.0, -5.0],
            'x8': [0.5, -0.5, 0.2, 0.1, 0.0],
            'x9': [5.4, 5.8, 6.2, 6.6, 7.0],
            'x10': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x11': [0.5, 1.5, 2.5, 3.5, 4.5],
            'City': ['Zaragoza', 'Albacete', 'Zaragoza', 'Albacete', 'Zaragoza'],
            'Country': ['Spain'] * 5,
            'target': [-10.0, 5.0, -12.0, 8.0, -15.0]
        })
        self.target = self.df_train.pop('target')

    def test_engineer_features(self):
        df_out = engineer_features(self.df_train, x5_median=10.0)
        
        # Test sentinel handling
        self.assertTrue((df_out['x5_is_missing'] == [0, 1, 0, 0, 1]).all())
        self.assertEqual(df_out['x5_imputed'].iloc[1], 10.0)
        
        # Test city encoding
        self.assertIn('City_encoded', df_out.columns)
        self.assertNotIn('City', df_out.columns)
        self.assertNotIn('Country', df_out.columns)
        self.assertEqual(df_out['City_encoded'].iloc[1], 1)
        self.assertEqual(df_out['City_encoded'].iloc[0], 0)
        
        # Test interaction
        self.assertIn('x4_x_x8', df_out.columns)
        
    def test_custom_preprocessor_fit_transform(self):
        preprocessor = PerfectFitPreprocessor()
        df_transformed, target = preprocessor.fit_transform(self.df_train.copy(), self.target)
        
        # Valid x5 values: 10.0, 8.0, 11.0 -> median is 10.0
        self.assertEqual(preprocessor.x5_median_, 10.0)
        
        # Test schema
        self.assertNotIn('City', df_transformed.columns)
        self.assertIn('City_encoded', df_transformed.columns)
        self.assertIn('x5_imputed', df_transformed.columns)

    def test_custom_preprocessor_transform(self):
        preprocessor = PerfectFitPreprocessor()
        preprocessor.fit_transform(self.df_train.copy(), self.target)
        
        df_test = pd.DataFrame({
            'x1': [0.1],
            'x2': [0.2],
            'x4': [0.1],
            'x5': [999.0], 
            'x6': [-5.0],
            'x7': [2.0],
            'x8': [0.5],
            'x9': [5.4],
            'x10': [1.0],
            'x11': [0.5],
            'City': ['Zaragoza'],
            'Country': ['Spain']
        })
        
        df_test_transformed, _ = preprocessor.transform(df_test)
        self.assertEqual(df_test_transformed['x5_imputed'].iloc[0], 10.0)
        self.assertEqual(df_test_transformed['City_encoded'].iloc[0], 0)

if __name__ == '__main__':
    unittest.main()
