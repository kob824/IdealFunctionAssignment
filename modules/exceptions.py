class TableNotFoundError(Exception):
    """Raised when a specified table does not exist in the database."""
    def __init__(self, table_name):
        super().__init__(f"Table '{table_name}' does not exist in the database.")

class ColumnMismatchError(Exception):
    """Raised when the columns in the CSV file do not match the table columns."""
    def __init__(self, table_name):
        super().__init__(f"Column mismatch detected for table '{table_name}'. Ensure the CSV headers match the table schema.")

class DataInsertionError(Exception):
    """Raised when there is an error inserting data into the database."""
    def __init__(self, table_name, message):
        super().__init__(f"Error inserting data into table '{table_name}': {message}")
