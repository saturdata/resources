#!/usr/bin/env python3
"""
Data Transformation in Python - Marimo Notebook
==============================================
Covers NumPy operations, Pandas vs Polars comparison, and data manipulation patterns
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data Transformation in Python
    Saturdata: Season 1

    This notebook focuses on fundamental data transformation operations including:
    - NumPy array operations and statistics
    - Pandas vs Polars performance and feature comparison
    - Data manipulation patterns and best practices
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    # Import required libraries for data transformation
    import time
    import numpy as np
    import pandas as pd
    import polars as pl
    import warnings

    warnings.filterwarnings("ignore")

    print("✅ Data transformation libraries imported successfully")
    return np, pd, pl, time


@app.cell
def _(mo):
    mo.md(r"""## NumPy basics""")
    return


@app.cell
def _(np):
    # Basic array creation and statistics
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Mean: {np.mean(data)}")
    print(f"Median: {np.median(data)}")
    print(f"Std Dev: {np.std(data)}")
    print(f"Variance: {np.var(data)}")
    return


@app.cell
def _(np):
    # Array operations and mathematical functions
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([2, 4, 6, 8, 10])

    print(f"Element-wise multiplication: {arr1 * arr2}")
    print(f"Dot product: {np.dot(arr1, arr2)}")
    print(f"Correlation coefficient: {np.corrcoef(arr1, arr2)[0,1]:.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Note
    The [0,1] is indexing into the correlation matrix that np.corrcoef() returns.
    When you call np.corrcoef(arr1, arr2), it doesn't just return a single correlation coefficient - it returns a 2x2 correlation matrix that looks like this:
    ```
    [[1.000, 0.xxx],
     [0.xxx, 1.000]]
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Polars data operations

    Polars is significantly faster than pandas because it's built with Rust and uses advanced techniques like parallel processing and lazy evaluation, which means it can handle large datasets much more efficiently. Unlike pandas, Polars has a more consistent and intuitive API that reduces common gotchas and makes your code more predictable and easier to debug. Additionally, Polars uses less memory and has better type safety, which helps prevent runtime errors and makes your data analysis more reliable, especially when working with production systems.
    """
    )
    return


@app.cell
def _(pl):
    # Create a Polars DataFrame
    df = pl.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 75000, 55000, 68000],
        'department': ['IT', 'Finance', 'IT', 'HR', 'Finance']
    })
    print(df)
    return (df,)


@app.cell
def _(df, pl):
    # Basic Polars operations
    summary_stats = df.select([
        pl.col('age').mean().alias('avg_age'),
        pl.col('salary').median().alias('median_salary'),
        pl.col('age').std().alias('age_std')
    ])
    print(summary_stats)
    return


@app.cell
def _(df, pl):
    # Group by operations in Polars
    dept_analysis = df.group_by('department').agg([
        pl.col('salary').mean().alias('avg_salary'),
        pl.col('age').max().alias('max_age'),
        pl.count().alias('employee_count')
    ])
    print(dept_analysis)
    return


@app.cell
def _(df, pl):
    # Filtering and transformations
    high_earners = df.filter(pl.col('salary') > 60000).with_columns([
        (pl.col('salary') * 1.1).alias('salary_with_bonus')
    ])
    print(high_earners)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Polars vs. pandas
    Data types and performance comparison
    """
    )
    return


@app.cell
def _(np):
    # Create test data with mixed types
    test_data = {
        'integers': [1, 2, 3, 4, 5] * 1000,
        'floats': [1.1, 2.2, 3.3, 4.4, 5.5] * 1000,
        'strings': ['A', 'B', 'C', 'D', 'E'] * 1000,
        'booleans': [True, False, True, False, True] * 1000,
        'dates': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 1000
    }

    # Demonstrate date parsing efficiency and type safety
    date_strings = ['2023-01-01', '2023-02-15', '2023-03-30'] * 10000

    # Demonstrate null handling differences
    data_with_nulls = {
        'mixed_ints': [1, 2, None, 4, 5],
        'mixed_floats': [1.1, None, 3.3, 4.4, 5.5],
        'mixed_strings': ['A', 'B', None, 'D', 'E']
    }

    # Create larger dataset for memory comparison
    large_data = {
        'id': range(100000),
        'category': ['A', 'B', 'C', 'D', 'E'] * 20000,
        'value': np.random.randn(100000),
        'flag': [True, False] * 50000
    }
    return data_with_nulls, date_strings, large_data, test_data


@app.cell
def _(pd, pl, test_data):

    print("=== DATA TYPE INFERENCE COMPARISON ===")

    # Pandas DataFrame
    pandas_df = pd.DataFrame(test_data)
    print("\nPandas data types:")
    print(pandas_df.dtypes)
    print(f"Pandas memory usage: {pandas_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Polars DataFrame
    polars_df = pl.DataFrame(test_data)
    print("\nPolars data types:")
    print(polars_df.dtypes)
    print(f"Polars estimated memory usage: {polars_df.estimated_size() / 1024**2:.2f} MB")
    return


@app.cell
def _(date_strings, pd, pl, time):
    print("\n=== DATE PARSING COMPARISON ===")

    # Pandas date parsing (requires explicit conversion)
    start_time = time.time()
    pandas_dates = pd.to_datetime(date_strings)
    pandas_time = time.time() - start_time
    print(f"Pandas date parsing time: {pandas_time:.4f} seconds")
    print(f"Pandas date type: {pandas_dates.dtype}")

    # Polars date parsing (automatic with proper schema)
    start_time = time.time()
    polars_dates = pl.Series("dates", date_strings).str.to_date()
    polars_time = time.time() - start_time
    print(f"Polars date parsing time: {polars_time:.4f} seconds")
    print(f"Polars date type: {polars_dates.dtype}")
    print(f"Polars is {pandas_time/polars_time:.1f}x faster for date parsing")
    return


@app.cell
def _(data_with_nulls, pd, pl):
    print("\n=== NULL HANDLING COMPARISON ===")

    # Pandas handling (promotes to object/float64)
    pandas_nulls = pd.DataFrame(data_with_nulls)
    print("Pandas with nulls:")
    print(pandas_nulls.dtypes)
    print("Note: Integers with nulls become float64 in pandas")

    # Polars handling (maintains types with proper null support)
    polars_nulls = pl.DataFrame(data_with_nulls)
    print("\nPolars with nulls:")
    print(polars_nulls.dtypes)
    print("Note: Polars maintains integer types with native null support")
    return


@app.cell
def _(large_data, pd, pl):
    # Memory efficiency comparison with large dataset
    print("\n=== MEMORY EFFICIENCY TEST ===")

    # Pandas memory usage
    pandas_large = pd.DataFrame(large_data)
    pandas_memory = pandas_large.memory_usage(deep=True).sum() / 1024**2

    # Polars memory usage
    polars_large = pl.DataFrame(large_data)
    polars_memory = polars_large.estimated_size() / 1024**2

    print(f"Pandas memory usage: {pandas_memory:.2f} MB")
    print(f"Polars memory usage: {polars_memory:.2f} MB")
    print(f"Memory savings with Polars: {(1 - polars_memory/pandas_memory)*100:.1f}%")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Business data example
    Combining NumPy, Polars for real-world analysis
    """
    )
    return


@app.cell
def _(np, pl):
    np.random.seed(42)

    # Generate business dataset
    business_data = pl.DataFrame({
        'sales': np.random.gamma(2, 1000, 500),
        'marketing_spend': np.random.uniform(100, 1000, 500),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
        'quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 500)
    })
    return (business_data,)


@app.cell
def _(business_data, pl):
    # Statistical summary with numpy and polars
    summary = business_data.select([
        pl.col('sales').mean().alias('avg_sales'),
        pl.col('sales').std().alias('sales_std'),
        pl.col('marketing_spend').median().alias('median_marketing')
    ])
    print("Business Data Summary:")
    print(summary)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook demonstrated:

    1. **NumPy Operations**: Fast numerical computations and array operations
    2. **Polars vs Pandas**: Performance, memory, and type system differences
    3. **Data Manipulation**: Group operations, filtering, and transformations
    4. **Real-world Applications**: Business data processing workflows

    Key insights:
    - Polars offers 2-5x performance improvements over pandas
    - Memory efficiency can save 20-30% in cloud computing costs  
    - Type safety prevents runtime errors in production systems
    - NumPy forms the foundation for all numerical operations
    """
    )
    return


if __name__ == "__main__":
    app.run()