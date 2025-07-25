"""
You may set up a MySQL database and use this script to profile its tables.

Following is the DDL and Data insertion:
-- Create the database
CREATE DATABASE IF NOT EXISTS my_profiling_db;

-- Use the database
USE my_profiling_db;

-- Create the users table
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100),
    registration_date DATE,
    age INT,
    salary DECIMAL(10, 2),
    status VARCHAR(20),
    description TEXT,
    is_active BOOLEAN
);

-- Create the transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INT PRIMARY KEY,
    user_id INT,
    amount DECIMAL(10, 2),
    transaction_date DATE,
    product_name VARCHAR(100),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Insert sample data into users
INSERT INTO users (id, username, email, registration_date, age, salary, status, description, is_active) VALUES
(1, 'alice', 'alice@example.com', '2023-01-15', 30, 60000.00, 'active', 'Software Engineer', TRUE),
(2, 'bob', 'bob@example.com', '2023-02-20', 25, 55000.50, 'active', 'Data Analyst', TRUE),
(3, 'charlie', 'charlie@example.com', '2023-03-01', 35, 70000.75, 'inactive', 'Project Manager', FALSE),
(4, 'diana', 'diana@example.com', '2023-04-10', 28, 62000.00, 'active', 'UX Designer', TRUE),
(5, 'eve', 'eve@example.com', '2023-05-05', 40, 80000.00, 'active', 'Team Lead', TRUE),
(6, 'frank', 'frank@example.com', '2023-06-11', NULL, 45000.00, 'active', 'Intern', TRUE),
(7, 'grace', NULL, '2023-07-01', 32, 68000.00, 'active', 'Consultant', TRUE),
(8, 'heidi', 'heidi@example.com', '2023-08-10', 29, 59000.00, 'inactive', 'Developer', FALSE),
(9, 'ivan', 'ivan@example.com', '2023-09-19', 45, 90000.00, 'active', 'Architect', TRUE),
(10, 'judy', 'judy@example.com', '2023-10-25', 27, 57000.00, 'active', 'QA Engineer', TRUE);

-- Insert sample data into transactions
INSERT INTO transactions (transaction_id, user_id, amount, transaction_date, product_name) VALUES
(101, 1, 150.75, '2023-01-20', 'Laptop'),
(102, 2, 50.00, '2023-02-25', 'Mouse'),
(103, 1, 200.00, '2023-03-05', 'Monitor'),
(104, 3, 30.25, '2023-04-12', 'Keyboard'),
(105, 4, 1000.00, '2023-05-08', 'Tablet'),
(106, 1, 120.00, '2024-01-01', 'Webcam'),
(107, 5, 75.50, '2024-02-10', 'Headphones');

"""

from typing import Any, Dict, List

import pymysql
from sqlalchemy import MetaData, Table, create_engine, func, inspect, types
from sqlalchemy.sql import select

# --- Configuration for MySQL Connection ---
# IMPORTANT: Replace with your MySQL server details
DB_USER = "root"  # Your MySQL username
DB_PASSWORD = "password"  # Your MySQL password
DB_HOST = "127.0.0.1"  # Or your MySQL server IP/hostname
DB_PORT = 3306  # MySQL default port
DB_NAME = "my_profiling_db"  # The database we created in MySQL Workbench

# SQLAlchemy connection string for MySQL
# 'mysql+mysqlconnector' is the recommended dialect for MySQL
DATABASE_URL = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# --- Data Profiling Function using SQLAlchemy ---
def profile_table_sqlalchemy_mysql(engine, table_name: str) -> Dict[str, Any]:
    """
    Profiles a given table in a MySQL database using SQLAlchemy.

    Args:
        engine: The SQLAlchemy engine connected to the database.
        table_name: The name of the table to profile.

    Returns:
        A dictionary containing profiling results for the table and its columns.
    """
    profiling_results: Dict[str, Any] = {}
    metadata = MetaData()

    # Reflect (introspect) the table's schema from the database
    try:
        table = Table(table_name, metadata, autoload_with=engine)
        profiling_results["table_name"] = table_name
    except Exception as e:
        print(f"Error reflecting table '{table_name}': {e}")
        return {"error": f"Could not find or reflect table '{table_name}'"}

    # Get total row count for percentages
    with engine.connect() as connection:
        total_rows_query = select(func.count()).select_from(table)
        total_rows = connection.execute(total_rows_query).scalar_one()
        profiling_results["total_rows"] = total_rows
        print(f"Profiling table '{table_name}' ({total_rows} rows)...")

        profiling_results["columns"] = {}

        for column in table.columns:
            column_name = column.name
            column_type = str(column.type)  # Get string representation of type
            column_profile: Dict[str, Any] = {
                "data_type": column_type,
                "nullable": column.nullable,
            }

            # --- Common Column Metrics ---
            # Non-null count
            non_null_count_query = select(func.count(column)).select_from(table)
            non_null_count = connection.execute(non_null_count_query).scalar_one()
            column_profile["non_null_count"] = non_null_count
            column_profile["null_count"] = total_rows - non_null_count
            column_profile["null_percentage"] = (
                (column_profile["null_count"] / total_rows * 100)
                if total_rows > 0
                else 0
            )

            # Distinct count
            distinct_count_query = select(
                func.count(func.distinct(column))
            ).select_from(table)
            distinct_count = connection.execute(distinct_count_query).scalar_one()
            column_profile["distinct_count"] = distinct_count
            column_profile["distinct_percentage"] = (
                (distinct_count / total_rows * 100) if total_rows > 0 else 0
            )

            # --- Type-Specific Profiling ---

            # Numerical Ranges (Integer, Numeric, Float)
            if isinstance(column.type, (types.Integer, types.Numeric, types.Float)):
                numeric_stats_query = (
                    select(
                        func.min(column).label("min_value"),
                        func.max(column).label("max_value"),
                        func.avg(column).label("avg_value"),
                        func.sum(column).label("sum_value"),
                        # Use func.stddev_samp for sample standard deviation (SQL standard)
                        # For older MySQL, 'STDDEV' might map to population. Test with your MySQL version.
                        func.stddev_samp(column).label("stddev_value"),
                    )
                    .select_from(table)
                    .where(column.isnot(None))
                )  # Exclude NULLs from calculations
                numeric_stats = connection.execute(numeric_stats_query).fetchone()

                if numeric_stats:
                    column_profile["numeric_stats"] = {
                        "min_value": numeric_stats.min_value,
                        "max_value": numeric_stats.max_value,
                        "avg_value": numeric_stats.avg_value,
                        "sum_value": numeric_stats.sum_value,
                        "stddev_value": numeric_stats.stddev_value,
                    }

            # Date Ranges (Date, DateTime, Timestamp)
            elif isinstance(column.type, (types.Date, types.DateTime, types.TIMESTAMP)):
                date_range_query = (
                    select(
                        func.min(column).label("min_date"),
                        func.max(column).label("max_date"),
                    )
                    .select_from(table)
                    .where(column.isnot(None))
                )  # Exclude NULLs
                date_range = connection.execute(date_range_query).fetchone()

                if date_range:
                    # Convert date objects to string for consistent output
                    column_profile["date_range"] = {
                        "min_date": (
                            str(date_range.min_date) if date_range.min_date else None
                        ),
                        "max_date": (
                            str(date_range.max_date) if date_range.max_date else None
                        ),
                    }

            # Text/String Metrics & Enumerations
            elif isinstance(column.type, types.String):
                string_length_query = (
                    select(
                        func.min(func.length(column)).label("min_length"),
                        func.max(func.length(column)).label("max_length"),
                        func.avg(func.length(column)).label("avg_length"),
                    )
                    .select_from(table)
                    .where(column.isnot(None))
                )  # Exclude NULLs from length calc
                string_lengths = connection.execute(string_length_query).fetchone()

                if string_lengths:
                    column_profile["string_lengths"] = {
                        "min_length": string_lengths.min_length,
                        "max_length": string_lengths.max_length,
                        "avg_length": string_lengths.avg_length,
                    }

                # Enumerations (unique values if distinct_percentage < 1%)
                if (
                    column_profile["distinct_percentage"] < 1.0
                    and column_profile["distinct_count"] > 0
                ):
                    enumeration_values_query = (
                        select(func.distinct(column))
                        .select_from(table)
                        .where(column.isnot(None))
                        .order_by(column)
                    )  # Order for consistency
                    enumeration_values = (
                        connection.execute(enumeration_values_query).scalars().all()
                    )
                    column_profile["is_enumeration"] = True
                    column_profile["enumeration_values"] = enumeration_values
                else:
                    column_profile["is_enumeration"] = False

            profiling_results["columns"][column_name] = column_profile

    return profiling_results


# --- Main Execution ---
if __name__ == "__main__":
    # Create the SQLAlchemy engine for MySQL
    try:
        db_engine = create_engine(DATABASE_URL)
        # Attempt to connect to verify credentials
        with db_engine.connect() as connection:
            connection.execute(select(1))
        print(f"Successfully connected to MySQL database: {DB_NAME}")
    except Exception as e:
        print(f"Error connecting to MySQL database: {e}")
        print(
            "Please check your DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, and DB_NAME configurations."
        )
        exit(1)

    inspector = inspect(db_engine)  # Inspector to get table names

    # Get all table names in the specified database
    all_table_names = inspector.get_table_names()
    if not all_table_names:
        print(
            f"No tables found in database '{DB_NAME}'. Please ensure tables are created and visible to the user."
        )
        exit(0)

    print(f"\nDiscovered tables in '{DB_NAME}': {all_table_names}\n")

    # Profile each discovered table
    all_profiling_output: Dict[str, Any] = {}
    for table_name in all_table_names:
        profile_data = profile_table_sqlalchemy_mysql(db_engine, table_name)
        all_profiling_output[table_name] = profile_data

        # --- Output Profiling Results ---
        if "error" in profile_data:
            print(f"Error profiling {table_name}: {profile_data['error']}\n")
            continue

        print(f"\n--- Profiling Results for Table: {profile_data['table_name']} ---")
        print(f"Total Rows: {profile_data['total_rows']}")

        for col_name, col_profile in profile_data["columns"].items():
            print(f"\n  Column: {col_name}")
            print(f"    Data Type: {col_profile['data_type']}")
            print(f"    Nullable: {col_profile['nullable']}")
            print(f"    Non-Null Count: {col_profile['non_null_count']}")
            print(f"    Null Count: {col_profile['null_count']}")
            print(f"    Null Percentage: {col_profile['null_percentage']:.2f}%")
            print(f"    Distinct Count: {col_profile['distinct_count']}")
            print(f"    Distinct Percentage: {col_profile['distinct_percentage']:.2f}%")

            if (
                "numeric_stats" in col_profile
                and col_profile["numeric_stats"]["min_value"] is not None
            ):
                print(f"    Numeric Stats:")
                print(f"      Min: {col_profile['numeric_stats']['min_value']}")
                print(f"      Max: {col_profile['numeric_stats']['max_value']}")
                print(f"      Avg: {col_profile['numeric_stats']['avg_value']:.2f}")
                print(f"      Sum: {col_profile['numeric_stats']['sum_value']:.2f}")
                print(
                    f"      StdDev: {col_profile['numeric_stats']['stddev_value']:.2f}"
                )
            elif (
                "date_range" in col_profile
                and col_profile["date_range"]["min_date"] is not None
            ):
                print(f"    Date Range:")
                print(f"      Min Date: {col_profile['date_range']['min_date']}")
                print(f"      Max Date: {col_profile['date_range']['max_date']}")
            elif (
                "string_lengths" in col_profile
                and col_profile["string_lengths"]["min_length"] is not None
            ):
                print(f"    String Lengths:")
                print(
                    f"      Min Length: {col_profile['string_lengths']['min_length']}"
                )
                print(
                    f"      Max Length: {col_profile['string_lengths']['max_length']}"
                )
                print(
                    f"      Avg Length: {col_profile['string_lengths']['avg_length']:.2f}"
                )

            if col_profile.get("is_enumeration"):
                print(f"    Enumeration Candidate (Unique values < 1%):")
                print(
                    f"      Values: {', '.join(map(str, col_profile['enumeration_values']))}"
                )
            # Optional: Show non-enumeration for clarity if needed
            # elif isinstance(col_profile['data_type'], str) and ('VARCHAR' in col_profile['data_type'].upper() or 'TEXT' in col_profile['data_type'].upper()):
            #      if col_profile['distinct_count'] > 0:
            #          print(f"    Not an Enumeration Candidate (Distinct values >= 1%)")

    print("\n--- Data Profiling with SQLAlchemy for MySQL Complete ---")
    # You can now further process `all_profiling_output`
    # For instance, save it to a JSON file, push it to a metadata catalog, etc.
