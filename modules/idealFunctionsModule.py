import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math

class BaseFitter(pd.DataFrame):  # Inheriting from pandas DataFrame class
    """
    Base class for fitting functions to data. Inherits from pandas DataFrame.

    Attributes:
        ideal_df (pd.DataFrame): DataFrame containing ideal functions.
        train_df (pd.DataFrame): DataFrame containing training data.
    """
    def __init__(self, ideal_df, train_df):
        """
        Initialize the BaseFitter with ideal and training data.

        Args:
            ideal_df (pd.DataFrame): DataFrame containing ideal functions.
            train_df (pd.DataFrame): DataFrame containing training data.
        """
        super().__init__()  # Initialize the parent class
        self.ideal_df = ideal_df
        self.train_df = train_df

class FunctionFitter(BaseFitter):
    """
    Class for fitting ideal functions to training data and selecting the best functions.

    Attributes:
        ideal_models (dict): Dictionary to store fitted models and their parameters.
        rmse_results (dict): Dictionary to store RMSE results for each model.
    """
    def __init__(self, ideal_df, train_df):
        """
        Initialize the FunctionFitter with ideal and training data.

        Args:
            ideal_df (pd.DataFrame): DataFrame containing ideal functions.
            train_df (pd.DataFrame): DataFrame containing training data.
        """
        super().__init__(ideal_df, train_df)  # Call the parent class constructor
        self.ideal_models = {}
        self.rmse_results = {}  # Dictionary to store RMSE results for each model

    def fit_ideal_functions(self):
        """
        Fit ideal functions to the training data and calculate RMSE for each combination.
        """
        train_x = self.train_df['x']
        train_x_const = sm.add_constant(train_x)  # Add a constant term to the predictor

        for col in self.ideal_df.columns:
            if col == 'x':
                continue  # Skip the x column

            x_values = self.ideal_df['x']
            y_values = self.ideal_df[col]
            x_const = sm.add_constant(x_values)
            model = sm.OLS(y_values, x_const).fit()  # Fit the model
            
            # Calculate the slope and intercept
            slope = model.params['x']
            intercept = model.params['const']

            # Prediction on the training x values using the ideal function model
            train_prediction = model.predict(train_x_const)

            # Calculate the Root Mean Squared Error using the training data
            rmse_dict = {}
            for train_col in self.train_df.columns:
                if train_col == 'x':
                    continue  # Skip the x column
                y_true = self.train_df[train_col]
                rmse = math.sqrt(mean_squared_error(y_true, train_prediction))
                rmse_dict[train_col] = rmse

            # Save model values and RMSE values
            self.ideal_models[col] = {'model': model, 'slope': slope, 'intercept': intercept}
            self.rmse_results[col] = rmse_dict

    def choose_best_ideal_functions(self):
        """
        Select the best ideal function for each training column based on RMSE.

        Returns:
            dict: Dictionary containing the best ideal function, slope, intercept, and RMSE for each training column.
        """
        best_functions = {}
        training_columns = [col for col in self.train_df.columns if col != 'x']  # Skip the x column
        for train_col in training_columns:
            best_rmse = float('inf')
            best_ideal = None
            for ideal_func, rmse_dict in self.rmse_results.items():
                if rmse_dict[train_col] < best_rmse:
                    best_rmse = rmse_dict[train_col]
                    best_ideal = ideal_func
            best_functions[train_col] = {
                'ideal_function': best_ideal,
                'slope': self.ideal_models[best_ideal]['slope'],
                'intercept': self.ideal_models[best_ideal]['intercept'],
                'rmse': best_rmse
            }
        return best_functions

class BaseMapper:
    """
    Base class for mapping test data to the best ideal functions.

    Attributes:
        best_functions (dict): Dictionary of best ideal functions and their parameters.
        test_df (pd.DataFrame): DataFrame containing test data.
        mapping_results (pd.DataFrame): DataFrame to store mapping results.
    """
    def __init__(self, best_functions, test_df):
        """
        Initialize the BaseMapper with best functions and test data.

        Args:
            best_functions (dict): Dictionary of best ideal functions and their parameters.
            test_df (pd.DataFrame): DataFrame containing test data.
        """
        self.best_functions = best_functions
        self.test_df = test_df
        self.mapping_results = pd.DataFrame(columns=['x', 'y', 'assigned_function', 'deviation'])

    def map_test_data(self):
        """
        Abstract method to map test data to ideal functions. Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class TestMapper(BaseMapper):
    """
    Class for mapping test data to the best ideal functions.

    Attributes:
        Inherits all attributes from BaseMapper.
    """
    def __init__(self, best_functions, test_df):
        """
        Initialize the TestMapper with best functions and test data.

        Args:
            best_functions (dict): Dictionary of best ideal functions and their parameters.
            test_df (pd.DataFrame): DataFrame containing test data.
        """
        super().__init__(best_functions, test_df)  # Call the parent class constructor

    def map_test_data(self): 
        """
        Map test data points to the best ideal functions based on minimum deviation.

        Returns:
            pd.DataFrame: DataFrame containing mapping results with columns ['x', 'y', 'assigned_function', 'deviation'].
        """
        results = []
        for _, row in self.test_df.iterrows():
            x_val = row['x']
            y_actual = row['y']
            best_error = float('inf')
            assigned_function = None
            # Loop through the best ideal functions.
            for train_col, func_info in self.best_functions.items():
                slope = func_info['slope']
                intercept = func_info['intercept']
                
                # Calculate the predicted y value using the ideal model
                y_pred = slope * x_val + intercept
                error = abs(y_actual - y_pred)
                if error < best_error:
                    best_error = error
                    assigned_function = func_info['ideal_function']
            results.append({
                'x': x_val,
                'y': y_actual,
                'assigned_function': assigned_function,
                'deviation': best_error
            })
        self.mapping_results = pd.DataFrame(results)
        return self.mapping_results