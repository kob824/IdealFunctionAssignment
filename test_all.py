import unittest
import pandas as pd
import math
from modules import idealfunctionsmodule

class TestFunctionFitter(unittest.TestCase):
    """
    Unit tests for the FunctionFitter class, which fits ideal functions to training data
    and selects the best functions based on RMSE.
    """

    def setUp(self):
        """
        Set up test data for FunctionFitter tests.
        Creates ideal and training DataFrames with known relationships.
        """
        self.ideal_df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'ideal1': [1, 3, 5, 7, 9],  # y = 2*x + 1
            'ideal2': [0, 3, 6, 9, 12]  # y = 3*x + 0
        })

        self.train_df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'train1': [1, 3, 5, 7, 9],  # y = 2*x + 1 (exact match with ideal1)
            'train2': [0, 4, 7, 10, 13]  # y = 3*x + 1 (not an exact match with ideal2)
        })

    def test_fit_and_choose_best_functions(self):
        """
        Test the fit_ideal_functions and choose_best_ideal_functions methods.
        Verifies that the ideal functions are fitted correctly and the best functions
        are chosen based on RMSE.
        """
        fitter = idealfunctionsmodule.FunctionFitter(self.ideal_df, self.train_df)
        fitter.fit_ideal_functions()
        
        self.assertIn('ideal1', fitter.ideal_models)
        self.assertIn('ideal2', fitter.ideal_models)
        
        # Checking if slope and intercept are calculated as expected
        ideal1_model = fitter.ideal_models['ideal1']
        ideal2_model = fitter.ideal_models['ideal2']
        self.assertAlmostEqual(ideal1_model['slope'], 2, places=2)
        self.assertAlmostEqual(ideal1_model['intercept'], 1, places=2)
        self.assertAlmostEqual(ideal2_model['slope'], 3, places=2)
        self.assertAlmostEqual(ideal2_model['intercept'], 0, places=2)
        
        # Get the best functions for the training data.
        best_functions = fitter.choose_best_ideal_functions()
        
        # For train1, the best ideal should be ideal1 (exact match).
        self.assertEqual(best_functions['train1']['ideal_function'], 'ideal1')
        
        # For train2, the best ideal should be ideal2.
        self.assertEqual(best_functions['train2']['ideal_function'], 'ideal2')

class TestTestMapper(unittest.TestCase):
    """
    Unit tests for the TestMapper class, which maps test data points to the best ideal functions
    based on minimum deviation.
    """

    def setUp(self):
        """
        Set up test data for TestMapper tests.
        Creates ideal, training, and test DataFrames with known relationships.
        """
        self.ideal_df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'ideal1': [1, 3, 5, 7, 9],
            'ideal2': [0, 3, 6, 9, 12]
        })
        self.train_df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'train1': [1, 3, 5, 7, 9],
            'train2': [0, 4, 7, 10, 13]
        })

        fitter = idealfunctionsmodule.FunctionFitter(self.ideal_df, self.train_df)
        fitter.fit_ideal_functions()
        self.best_functions = fitter.choose_best_ideal_functions()
        
        # Create a test dataframe.
        # For x = 2:
        #   ideal1 predicts 2*2+1 = 5, ideal2 predicts 3*2+0 = 6.
        # For x = 3:
        #   ideal1 predicts 2*3+1 = 7, ideal2 predicts 3*3+0 = 9.
        # We choose y values to clearly favor one model over the other.
        self.test_df = pd.DataFrame({
            'x': [2, 3],
            'y': [5, 9]  # Row 1 should map to ideal1, row 2 should map to ideal2.
        })
    
    def test_map_test_data(self):
        """
        Test the map_test_data method to ensure correct mapping of test data to ideal functions.
        Verifies that the assigned functions and deviations are calculated correctly.
        """
        mapper = idealfunctionsmodule.TestMapper(self.best_functions, self.test_df)
        mapping_results = mapper.map_test_data()
        
        # Check that the results contain the expected columns.
        self.assertTrue('x' in mapping_results.columns)
        self.assertTrue('y' in mapping_results.columns)
        self.assertTrue('assigned_function' in mapping_results.columns)
        self.assertTrue('deviation' in mapping_results.columns)
        
        # Verify that there are exactly two rows.
        self.assertEqual(len(mapping_results), 2)
        
        # For the first row (x = 2, y = 5), ideal1 should be chosen.
        row1 = mapping_results.iloc[0]
        self.assertEqual(row1['assigned_function'], 'ideal1')
        self.assertAlmostEqual(row1['deviation'], 0, places=5)
        
        # For the second row (x = 3, y = 9), ideal2 should be chosen.
        row2 = mapping_results.iloc[1]
        self.assertEqual(row2['assigned_function'], 'ideal2')
        self.assertAlmostEqual(row2['deviation'], 0, places=5)

if __name__ == '__main__':
    """
    Run all unit tests.
    """
    unittest.main()
