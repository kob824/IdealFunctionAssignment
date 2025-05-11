from modules import idealfunctionsmodule
from modules import plotdata
from modules import sqlite_helper
import pandas as pd

def get_pd_from_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path)

def get_data_from_source(source_method=3, csv_path_prefix="data/", db_path="data/data.db"):
    """
    Returns dataframes based on the specified data source method.
    
    Args:
        source_method (int): Method to get data:
            1 - Load directly from CSV files
            2 - Insert CSV data into SQLite and then retrieve it
            3 - Retrieve data from existing SQLite database (default)
        csv_path_prefix (str): Path prefix for CSV files (default: "data/")
        db_path (str): Path to SQLite database (default: "data/data.db")
    
    Returns:
        tuple: (ideal_df, train_df, test_df) containing the datasets
    """
    if source_method == 1:
        print("Loading data directly from CSV files...")
        ideal_df = get_pd_from_csv(f"{csv_path_prefix}ideal.csv")
        train_df = get_pd_from_csv(f"{csv_path_prefix}train.csv")
        test_df = get_pd_from_csv(f"{csv_path_prefix}test.csv")
    elif source_method == 2:
        print("Inserting CSV data into SQLite and retrieving it...")
        sqlite_helper.insert_csv_to_table(f"{csv_path_prefix}ideal.csv", db_path, "ideal")
        sqlite_helper.insert_csv_to_table(f"{csv_path_prefix}train.csv", db_path, "train")
        sqlite_helper.insert_csv_to_table(f"{csv_path_prefix}test.csv", db_path, "test")
        ideal_df = sqlite_helper.get_table_data_as_df(db_path, "ideal")
        train_df = sqlite_helper.get_table_data_as_df(db_path, "train")
        test_df = sqlite_helper.get_table_data_as_df(db_path, "test")
    else:  # Default to option 3 or handle any invalid input
        print("Retrieving data from existing SQLite database...")
        ideal_df = sqlite_helper.get_table_data_as_df(db_path, "ideal")
        train_df = sqlite_helper.get_table_data_as_df(db_path, "train")
        test_df = sqlite_helper.get_table_data_as_df(db_path, "test")
    
    return ideal_df, train_df, test_df

def main():
    """
    Main function to execute the workflow:
    1. Load ideal, training, and test datasets.
    2. Fit ideal functions to training data.
    3. Choose the best ideal functions based on RMSE.
    4. Map test data to the best ideal functions.
    5. Generate and display plots for ideal, training, and test data.
    """
    # Get data using the default source method (3)
    ideal_df, train_df, test_df = get_data_from_source()

    fitter = idealfunctionsmodule.FunctionFitter(ideal_df, train_df)
    fitter.fit_ideal_functions()

    best_functions = fitter.choose_best_ideal_functions()
    print("Chosen ideal functions based on training RMSE:")
    for train_col, info in best_functions.items():
        print(f"{train_col} -> Ideal Function: {info['ideal_function']}, RMSE: {info['rmse']:.4f}")

    mapper = idealfunctionsmodule.TestMapper(best_functions, test_df)
    mapping_results = mapper.map_test_data()
    print("Test Data Mapping Results:")
    print(mapping_results)

    plotdata.plot_ideal_functions(ideal_df)
    plotdata.plot_training_data(train_df)
    plotdata.plot_test_data_with_ideal(test_df, ideal_df, best_functions)
    print("Plots generated successfully.")

if __name__ == "__main__":
    """
    Entry point of the script. Calls the main function.
    """
    main()