import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math

class FunctionFitter:
    def __init__(self, ideal_df, train_df):
        self.ideal_df = ideal_df
        self.train_df = train_df
        self.ideal_models = {}
        self.rmse_results = {} # Dictionary to store RMSE results for each model where key is the ideal function column name and value is the dict of RMSE results


    def fit_ideal_functions(self):
        train_x = self.train_df['x']
        train_x_const = sm.add_constant(train_x)  # Add a constant term to the predictor

        for col in self.ideal_df.columns:
            if col == 'x':
                continue # Skip the x column

            x_values = self.ideal_df['x']
            y_values = self.ideal_df[col]
            x_const = sm.add_constant(x_values) 
            model = sm.OLS(y_values, x_const).fit()  #This method from statsmodels.api is used to fit the model
            
            # Now calculate the slope and intercept
            slope = model.params['x']
            intercept = model.params['const']

            #Predioction on the training x values using the ideal function model
            train_prediction = model.predict(train_x_const)

            # Calculate the Root Mean Squared Error using the training data
            rmse_dict = {}
            for train_col in self.train_df.columns:
                if train_col == 'x':
                    continue # Skip the x column
                y_true = self.train_df[train_col]
                rmse = math.sqrt(mean_squared_error(y_true, train_prediction))
                rmse_dict[train_col] = rmse


            #Save model values and RMSE values in case we do use them later
            self.ideal_models[col] = {'model': model, 'slope': slope, 'intercept': intercept}
            self.rmse_results[col] = rmse_dict

    def choose_best_ideal_functions(self):
        best_functions = {}
        training_columns = [col for col in self.train.df.columns if col != 'x'] # skipping the x column again
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
