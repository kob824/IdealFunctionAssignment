from sqlalchemy import create_engine, inspect
import pandas as pd

try:
    # When running from the main module (e.g., main.py)
    from modules import exceptions as ex
except ImportError:
    # When running directly from this module
    import exceptions as ex

def insert_csv_to_table(csv_file_path, db_path):
    """
    Inserts data from a CSV file into a SQLite table if the table exists.
    The table name is derived from the CSV file name (without extension).
    Headers in the CSV file must match the table columns.

    Args:
        csv_file_path (str): Path to the CSV file.
        db_path (str): Path to the SQLite database.
    """
    table_name = csv_file_path.split('/')[-1].split('.')[0]

    engine = create_engine(f'sqlite:///{db_path}')

    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise ex.TableNotFoundError(table_name)

    df = pd.read_csv(csv_file_path)

    try:
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Data from '{csv_file_path}' inserted into table '{table_name}' successfully.")
    except Exception as e:
        raise ex.DataInsertionError(table_name, str(e))


def get_table_data_as_df(db_path, table_name):
    """
    Retrieves data from a specified SQLite table and returns it as a pandas DataFrame,
    excluding the 'id' column if it exists.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to retrieve data from.

    Returns:
        pd.DataFrame: DataFrame containing the data from the specified table, excluding the 'id' column.
    """
    # Create a database engine
    engine = create_engine(f'sqlite:///{db_path}')

    # Load data from the specified table into a DataFrame
    df = pd.read_sql_table(table_name, engine)

    # Drop the 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    return df


# Small test block to check if the exceptions work

# if __name__ == "__main__":
#     try:
#         insert_csv_to_table("non_existent.csv", "test.db")
#     except ex.TableNotFoundError as e:
#         print(e)

#     try:
#         insert_csv_to_table("data/ideal.csv", "test.db")
#     except ex.DataInsertionError as e:
#         print(e)
