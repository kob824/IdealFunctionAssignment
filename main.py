from modules import idealfunctionsmodule
from modules import plotdata
import pandas as pd

def get_pd_from_csv(file_path):
    return pd.read_csv(file_path)

def main():
    ideal_df = get_pd_from_csv("data/ideal.csv")
    train_df = get_pd_from_csv("data/train.csv")
    test_df = get_pd_from_csv("data/test.csv")

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
    main()