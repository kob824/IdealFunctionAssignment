import matplotlib.pyplot as plt
import numpy as np


def plot_training_data(train_df):
    """
    Plots the training data.
    """
    plt.figure(figsize=(10, 6))
    for col in train_df.columns:
        if col == 'x':
            continue
        plt.plot(train_df['x'], train_df[col], linestyle='--', label=f"Train {col}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Training Data")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ideal_functions(ideal_df):
    """
    Plots all ideal functions.
    """
    plt.figure(figsize=(10, 6))
    for col in ideal_df.columns:
        if col == 'x':
            continue
        plt.plot(ideal_df['x'], ideal_df[col], label=f"Ideal {col}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ideal Functions")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_test_data_with_ideal(test_df, ideal_df, best_functions):
    """
    Plots the test data along with the chosen ideal functions.
    """
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'm']
    x_vals = ideal_df['x']
    i = 0
    for train_col, func_info in best_functions.items():
        ideal_func = func_info['ideal_function']
        y_vals = ideal_df[ideal_func]
        plt.plot(x_vals, y_vals, color=colors[i % len(colors)],
                 linewidth=2, label=f"Chosen Ideal {ideal_func} for {train_col}")
        i += 1
    plt.scatter(test_df['x'], test_df['y'], color='black', label="Test Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Test Data and Chosen Ideal Functions")
    plt.legend()
    plt.grid(True)
    plt.show()