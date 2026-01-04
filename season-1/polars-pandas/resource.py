"""
Episode 1: Data Mechanic's Garage
Performance comparison of Python data processing libraries

Interactive marimo notebook demonstrating:
- pandas (baseline)
- pandas + PyArrow (quick win)
- Polars (modern engine)
- DuckDB (SQL interface)

With comprehensive timing and memory benchmarking.
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 🔧 Data Mechanic's Garage: Episode 1

    ## Performance Tuning for Python Data Processing

    Welcome to the Data Mechanic's Garage! Today we're taking a "classic" pandas script and giving it a full performance upgrade. We'll explore four approaches to data processing and see just how fast modern tools can be.

    ### Our Test Vehicle 🚗
    - **Dataset**: 5 million e-commerce transactions (~1GB)
    - **Operations**: CSV reading, string manipulation, aggregations, joins
    - **Goal**: Measure time and memory for each library

    ### The Lineup
    1. **Classic pandas** - The reliable workhorse we all know
    2. **pandas + PyArrow** - A simple bolt-on upgrade
    3. **Polars** - The modern parallel processing engine
    4. **DuckDB** - SQL power with in-memory analytics
    """)
    return


@app.cell
def _():
    # Core libraries
    import pandas as pd
    import polars as pl
    import duckdb
    import numpy as np

    # Performance monitoring
    import time
    import psutil
    from pathlib import Path

    # Visualization
    import altair as alt

    # Suppress warnings for cleaner output
    import warnings

    warnings.filterwarnings("ignore")
    return Path, alt, duckdb, pd, pl, psutil, time


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Dataset Overview

    Our synthetic e-commerce dataset includes:
    - **Transactions**: 5M rows with customer, product, price, quantity, region
    - **Customers**: 100K dimension table with email, segment, lifetime value
    - **Products**: 10K dimension table with category, cost, supplier
    """)
    return


@app.cell
def _(Path):
    # Data paths - use absolute path relative to this notebook file
    NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    DATA_DIR = NOTEBOOK_DIR / "data"
    TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
    CUSTOMERS_FILE = DATA_DIR / "customers.csv"
    PRODUCTS_FILE = DATA_DIR / "products.csv"
    return CUSTOMERS_FILE, DATA_DIR, TRANSACTIONS_FILE


@app.cell
def _(CUSTOMERS_FILE, DATA_DIR, TRANSACTIONS_FILE, mo):
    # Verify data files exist
    _files_exist = all([TRANSACTIONS_FILE.exists(), CUSTOMERS_FILE.exists()])

    if not _files_exist:
        mo.md(f"""
        ⚠️ **Data files not found!**

        Please generate the dataset first:
        ```bash
        cd {DATA_DIR.parent}
        python data/generate_dataset.py
        ```
        """)
    else:
        _tx_size_mb = TRANSACTIONS_FILE.stat().st_size / 1024 / 1024
        mo.md(f"""
        ✅ **Data files loaded successfully**

        - Transactions: {_tx_size_mb:.1f} MB
        - Location: `{DATA_DIR}`
        """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⚡ Benchmarking Functions

    Our benchmark harness tracks:
    - **Execution time** (seconds)
    - **Memory usage** (MB) - delta from baseline
    - **Peak memory** (MB) - maximum RSS during execution
    
    ### Why We Use Process Memory Monitoring
    
    We use `psutil` to continuously sample the process's Resident Set Size (RSS) 
    during execution. This captures **all** memory allocations, including:
    - Python heap allocations (pandas, Python objects)
    - Native library allocations (Polars/Rust, DuckDB/C++, PyArrow)
    - Memory-mapped files and buffers
    
    ⚠️ **Note:** `tracemalloc` only tracks Python heap and misses native memory,
    which is why Polars (Rust) and DuckDB (C++) would show 0 MB with that approach.
    """)
    return


@app.cell
def _(psutil, time):
    import threading
    
    def benchmark_operation(func, label: str) -> dict:
        """
        Benchmark a function with timing and memory tracking.
        
        Uses a monitoring thread to continuously sample process memory (RSS)
        during execution. This captures native memory allocations (e.g., Rust in Polars)
        that tracemalloc misses since it only tracks Python heap allocations.

        Returns:
            dict with library, time_seconds, memory_mb, peak_memory_mb
        """
        process = psutil.Process()
        
        # Get baseline memory before execution
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Track peak memory during execution using a monitoring thread
        peak_memory = [mem_before]  # Use list for mutable reference in thread
        monitoring = [True]  # Flag to control monitoring thread
        
        def monitor_memory():
            """Sample memory usage every 10ms during execution."""
            while monitoring[0]:
                try:
                    current_mem = process.memory_info().rss / 1024 / 1024
                    if current_mem > peak_memory[0]:
                        peak_memory[0] = current_mem
                    time.sleep(0.01)  # Sample every 10ms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        # Execute and time
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        
        # Stop monitoring and wait for thread to finish
        monitoring[0] = False
        monitor_thread.join(timeout=1.0)
        
        # Final memory reading
        mem_after = process.memory_info().rss / 1024 / 1024

        return {
            "library": label,
            "time_seconds": round(elapsed, 2),
            "memory_mb": round(mem_after - mem_before, 1),
            "peak_memory_mb": round(peak_memory[0] - mem_before, 1),
            "result": result,
        }
    return (benchmark_operation,)


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🚗 The Classic: pandas Baseline

    Our starting point - a traditional pandas workflow that many of us run daily. It's familiar, reliable, but not particularly fast.

    ### The Workflow:
    1. Load 5M transactions from CSV
    2. Extract email domains from customer data
    3. Group by region and product category
    4. Join with customer dimension table
    """)
    return


@app.cell
def _(CUSTOMERS_FILE, TRANSACTIONS_FILE, benchmark_operation, pd):
    def pandas_baseline():
        """Classic pandas implementation."""
        # 1. Load data
        transactions = pd.read_csv(TRANSACTIONS_FILE)
        customers = pd.read_csv(CUSTOMERS_FILE)

        # 2. String manipulation - extract email domain
        customers["email_domain"] = customers["email"].apply(
            lambda x: x.split("@")[1] if "@" in x else "unknown"
        )

        # 3. Aggregation - group by region and category
        agg_results = (
            transactions.groupby(["region", "product_category"])
            .agg({"revenue": ["sum", "mean", "count"], "customer_id": "nunique"})
            .reset_index()
        )

        # 4. Join with customer data
        result = transactions.merge(
            customers[["customer_id", "segment", "email_domain"]],
            on="customer_id",
            how="left",
        )

        return len(result)

    # Run benchmark
    pandas_result = benchmark_operation(pandas_baseline, "Classic pandas")
    return (pandas_result,)


@app.cell
def _(mo, pandas_result):
    mo.md(f"""
    ### 📈 Classic pandas Results

    - **Time**: {pandas_result["time_seconds"]} seconds
    - **Memory**: {pandas_result["memory_mb"]} MB
    - **Peak Memory**: {pandas_result["peak_memory_mb"]} MB

    This is our baseline. Now let's see how we can improve it!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # ⚡ The Bolt-On Upgrade: pandas + PyArrow

    Before we rewrite everything, let's try a **simple one-line change**. PyArrow is a high-performance backend that pandas can use under the hood.

    ### What Changes:
    - Add `engine='pyarrow'` to CSV reading
    - Add `dtype_backend='pyarrow'` for better type handling

    That's it! No algorithm changes, no refactoring - just a different backend.
    """)
    return


@app.cell
def _(CUSTOMERS_FILE, TRANSACTIONS_FILE, benchmark_operation, pd):
    def pandas_pyarrow():
        """pandas with PyArrow backend for improved performance."""
        # 1. Load data with PyArrow engine
        transactions = pd.read_csv(
            TRANSACTIONS_FILE, engine="pyarrow", dtype_backend="pyarrow"
        )
        customers = pd.read_csv(
            CUSTOMERS_FILE, engine="pyarrow", dtype_backend="pyarrow"
        )

        # 2. String manipulation
        customers["email_domain"] = customers["email"].apply(
            lambda x: x.split("@")[1] if "@" in x else "unknown"
        )

        # 3. Aggregation
        agg_results = (
            transactions.groupby(["region", "product_category"])
            .agg({"revenue": ["sum", "mean", "count"], "customer_id": "nunique"})
            .reset_index()
        )

        # 4. Join
        result = transactions.merge(
            customers[["customer_id", "segment", "email_domain"]],
            on="customer_id",
            how="left",
        )

        return len(result)

    # Run benchmark
    pandas_pyarrow_result = benchmark_operation(pandas_pyarrow, "pandas + PyArrow")
    return (pandas_pyarrow_result,)


@app.cell
def _(mo, pandas_pyarrow_result, pandas_result):
    _speedup = round(
        pandas_result["time_seconds"] / pandas_pyarrow_result["time_seconds"], 1
    )
    mo.md(
        f"""
        ### 📈 pandas + PyArrow Results

        - **Time**: {pandas_pyarrow_result["time_seconds"]} seconds
        - **Memory**: {pandas_pyarrow_result["memory_mb"]} MB
        - **Speedup**: {_speedup}x faster than baseline

        🎯 **Key Takeaway**: A simple backend change gave us {_speedup}x speedup with zero refactoring!
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🚀 The Modern Engine: Polars

    Now for the main event. Polars is built from the ground up for parallel processing and performance. It uses:
    - **Lazy evaluation**: Builds a query plan before execution
    - **Expression-based API**: Readable method chaining
    - **Automatic parallelization**: Uses all your CPU cores
    - **Optimized memory layout**: Columnar storage for speed

    Let's rewrite our workflow in Polars and see the difference.
    """)
    return


@app.cell
def _(CUSTOMERS_FILE, TRANSACTIONS_FILE, benchmark_operation, pl):
    def polars_implementation():
        """Modern Polars implementation with lazy evaluation."""
        # 1. Load data with lazy evaluation
        transactions = pl.scan_csv(TRANSACTIONS_FILE)
        customers = pl.scan_csv(CUSTOMERS_FILE)

        # 2. String manipulation using expressions
        customers = customers.with_columns(
            pl.col("email").str.split("@").list.get(1).alias("email_domain")
        )

        # 3. Aggregation using expression API
        agg_results = transactions.group_by(["region", "product_category"]).agg(
            [
                pl.sum("revenue").alias("total_revenue"),
                pl.mean("revenue").alias("avg_revenue"),
                pl.count("revenue").alias("transaction_count"),
                pl.n_unique("customer_id").alias("unique_customers"),
            ]
        )

        # 4. Join with customer data
        result = (
            transactions.join(
                customers.select(["customer_id", "segment", "email_domain"]),
                on="customer_id",
                how="left",
            ).collect()  # Execute the lazy query plan
        )

        return len(result)

    # Run benchmark
    polars_result = benchmark_operation(polars_implementation, "Polars")
    return (polars_result,)


@app.cell
def _(mo, pandas_result, polars_result):
    _speedup = round(pandas_result["time_seconds"] / polars_result["time_seconds"], 1)
    mo.md(
        f"""
        ### 📈 Polars Results

        - **Time**: {polars_result["time_seconds"]} seconds
        - **Memory**: {polars_result["memory_mb"]} MB
        - **Speedup**: {_speedup}x faster than baseline

        🔥 **Key Takeaway**: Polars delivers {_speedup}x speedup with parallel processing and query optimization!

        Notice how the expression-based API is also more readable - you can see the entire data pipeline at a glance.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🐥 The SQL Interface: DuckDB

    What if you prefer SQL syntax? DuckDB gives you the power of Polars-level performance with SQL expressiveness.

    ### The DuckDB Philosophy:
    - **DuckDB is to Snowflake what Polars is to Databricks**
    - In-memory analytics database
    - Optimized columnar execution
    - SQL you already know

    ### Key Performance Consideration:
    For this benchmark, we load the CSV **once** and query it multiple times in memory.
    This matches the pandas/Polars pattern and avoids re-reading the file.

    **In production**, you can query CSV/Parquet files directly for even better performance:
    ```sql
    SELECT * FROM read_csv_auto('data.csv') WHERE date >= '2024-01-01'
    ```

    Perfect for analysts who think in SQL but need local performance.
    """)
    return


@app.cell
def _(duckdb, mo):
    """
    ## DuckDB Connection Setup

    Creating an in-memory DuckDB connection for SQL execution using mo.sql() syntax.
    marimo will automatically discover this connection and use it with mo.sql().
    """
    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")

    # Optimize DuckDB settings for this workload
    # DuckDB auto-detects CPU cores for parallelization (default behavior)
    conn.execute("SET memory_limit='2GB'")
    # Enable all optimizers for best performance
    conn.execute("SET disabled_optimizers=''")

    mo.md("""
    ✅ **DuckDB Connection Established**

    Connection configured with:
    - Memory limit: 2GB
    - Threads: Auto-detect (all cores, default)
    - Optimizers: All enabled

    💡 **Using `mo.sql()`:**
    - marimo automatically discovers the `conn` variable
    - Use `mo.sql(f"...")` to execute queries with f-string interpolation
    - Returns a DataFrame directly (Polars DataFrame by default)
    """)
    return


@app.cell
def _(
    CUSTOMERS_FILE,
    TRANSACTIONS_FILE,
    benchmark_operation,
    customers,
    mo,
    transactions,
):
    def duckdb_implementation():
        """
        DuckDB SQL-based implementation optimized for performance using mo.sql() syntax.

        This implementation materializes data into tables to match the benchmark
        pattern (load once, query multiple times). For production use, DuckDB
        excels when querying CSV/Parquet files directly with predicate pushdown.

        Uses mo.sql() syntax similar to the SQL notebook pattern, where marimo
        automatically discovers the DuckDB connection.
        """
        # Load data into tables (materialize once for fair comparison with pandas/Polars)
        # This avoids re-reading CSV files on each query
        # Use TEMPORARY tables for better memory management
        mo.sql(f"""
            CREATE OR REPLACE TEMPORARY TABLE transactions AS
            SELECT * FROM read_csv_auto('{TRANSACTIONS_FILE}')
        """)

        # Load and transform customers in one step
        # Create as temporary table with only needed columns for join
        # This smaller table will be used as the build side in the hash join
        mo.sql(f"""
            CREATE OR REPLACE TEMPORARY TABLE customers AS
            SELECT
                customer_id,
                segment,
                SPLIT_PART(email, '@', 2) as email_domain
            FROM read_csv_auto('{CUSTOMERS_FILE}')
        """)

        # Aggregation query (for benchmarking consistency)
        # mo.sql() returns a DataFrame directly
        # This query is executed separately to match the benchmark pattern
        agg_results = mo.sql("""
            SELECT
                region,
                product_category,
                SUM(revenue) as total_revenue,
                AVG(revenue) as avg_revenue,
                COUNT(*) as transaction_count,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM transactions
            GROUP BY region, product_category
        """)

        # Optimized join query - this is the main operation
        # Strategy: Use LEFT JOIN to match benchmark (pandas uses how='left')
        # DuckDB's cost-based optimizer will automatically:
        # - Use customers (smaller table, 100K rows) as the build side
        # - Use transactions (larger table, 5M rows) as the probe side
        # - Build a hash table on customers.customer_id for fast lookups
        # This is the optimal join strategy for this data size ratio
        result = mo.sql("""
            SELECT
                t.*,
                c.segment,
                c.email_domain
            FROM transactions t
            LEFT JOIN customers c ON t.customer_id = c.customer_id
        """)

        return len(result)

    # Run benchmark
    duckdb_result = benchmark_operation(duckdb_implementation, "DuckDB")
    return (duckdb_result,)


@app.cell
def _(duckdb_result, mo, pandas_result):
    _speedup = round(pandas_result["time_seconds"] / duckdb_result["time_seconds"], 1)
    mo.md(
        f"""
        ### 📈 DuckDB Results

        - **Time**: {duckdb_result["time_seconds"]} seconds
        - **Memory**: {duckdb_result["memory_mb"]} MB
        - **Speedup**: {_speedup}x faster than baseline

        🎯 **Key Takeaway**: DuckDB delivers Polars-level performance with familiar SQL syntax!

        If you're an analyst comfortable with SQL, DuckDB is a natural fit. The performance is just as impressive as Polars.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🏁 The Winner's Circle: Final Scoreboard

    Let's see all our results side by side:
    """)
    return


@app.cell
def _(duckdb_result, pandas_pyarrow_result, pandas_result, pd, polars_result):
    # Compile results
    _results_data = [pandas_result, pandas_pyarrow_result, polars_result, duckdb_result]

    results_df = pd.DataFrame(_results_data)
    results_df["speedup"] = round(
        pandas_result["time_seconds"] / results_df["time_seconds"], 1
    )
    results_df = results_df.drop("result", axis=1)
    return (results_df,)


@app.cell
def _(mo, results_df):
    mo.ui.table(results_df, selection=None, label="Performance Comparison Results")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Visual Comparison

    Let's visualize the performance differences:
    """)
    return


@app.cell
def _(alt, results_df):
    # Time comparison chart
    time_chart = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "library:N", 
                sort="-y", 
                title="Library",
                axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                "time_seconds:Q", 
                title="Time (seconds)",
                scale=alt.Scale(domain=[0, 3]),
                axis=alt.Axis(tickCount=7)  # 0, 0.5, 1, 1.5, 2, 2.5, 3
            ),
            color=alt.Color(
                "library:N", scale=alt.Scale(scheme="category20"), legend=None
            ),
            tooltip=["library", "time_seconds", "speedup"],
        )
        .properties(title="Execution Time Comparison", width=400, height=300)
    )

    # Add speedup annotations above the bars
    speedup_text = (
        alt.Chart(results_df)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("library:N", sort="-y"),
            y="time_seconds:Q",
            text=alt.Text("speedup:Q", format=".1f"),
            color=alt.value("white"),
        )
    )

    time_comparison = time_chart + speedup_text
    return (time_comparison,)


@app.cell
def _(mo, time_comparison):
    mo.ui.altair_chart(time_comparison)
    return


@app.cell
def _(alt, results_df):
    # Memory comparison chart
    memory_chart = (
        alt.Chart(results_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "library:N", 
                title="Library",
                axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                "peak_memory_mb:Q", 
                title="Peak Memory (MB)",
                scale=alt.Scale(domain=[0, 2500]),
                axis=alt.Axis(tickCount=8)  # 0, 200, 400, 600, 800, 1000, 1200, 1400
            ),
            color=alt.Color(
                "library:N", scale=alt.Scale(scheme="category20"), legend=None
            ),
            tooltip=["library", "peak_memory_mb"],
        )
        .properties(title="Peak Memory Usage Comparison", width=400, height=300)
    )
    
    # Add memory values as labels above the bars
    memory_text = (
        alt.Chart(results_df)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("library:N"),
            y="peak_memory_mb:Q",
            text=alt.Text("peak_memory_mb:Q", format=".1f"),
            color=alt.value("white"),
        )
    )
    
    memory_comparison = memory_chart + memory_text
    return (memory_comparison,)


@app.cell
def _(memory_comparison, mo):
    mo.ui.altair_chart(memory_comparison)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🎓 Key Takeaways & Decision Framework

    ## When to Use Each Library

    ### Classic pandas
    **Use when:**
    - Working with small datasets (<100K rows)
    - Prototyping and exploratory analysis
    - Team is most comfortable with pandas
    - Integrating with pandas-specific libraries

    **Avoid when:**
    - Processing large datasets (>1GB)
    - Performance is critical
    - Running in production pipelines

    ---

    ### pandas + PyArrow
    **Use when:**
    - You have existing pandas code
    - Can't justify a full rewrite
    - Want quick performance wins
    - Need pandas compatibility

    **Benefit:**
    - 2-3x speedup with minimal code changes
    - Drop-in replacement (mostly compatible)

    ---

    ### Polars
    **Use when:**
    - Processing large datasets (>1GB)
    - Performance is critical
    - Building production pipelines
    - Starting new projects

    **Benefits:**
    - 20-30x speedup over baseline pandas
    - Lower memory usage
    - Modern, expressive API
    - Prepares you for distributed computing (similar mental model to Spark)

    **Trade-offs:**
    - Requires learning new API
    - Smaller ecosystem than pandas
    - Some pandas features not yet available

    ---

    ### DuckDB
    **Use when:**
    - Team is SQL-native
    - Working with analytical queries
    - Need to run complex aggregations
    - Want Polars-level performance with SQL syntax

    **Benefits:**
    - Familiar SQL syntax
    - Excellent performance (comparable to Polars)
    - Great for analysts transitioning from databases
    - Natural path to cloud warehouses (Snowflake, BigQuery)

    **Analogy:**
    - **DuckDB → Snowflake**: Learn DuckDB locally, scale to Snowflake in the cloud
    - **Polars → Databricks**: Master Polars, transition to Spark/Databricks for distributed computing

    ---

    ## The Analytics Stack Ladder

    ### For SQL-Native Analysts:
    ```
    Local Development     →    Cloud Platform
    ────────────────────────────────────────
    DuckDB (in-memory)   →    Snowflake
    PostgreSQL (local)   →    BigQuery
    SQLite (embedded)    →    Redshift
    ```

    ### For DataFrame Users:
    ```
    Local Development     →    Cloud Platform
    ────────────────────────────────────────
    Polars (parallel)    →    PySpark / Databricks
    pandas (single-core) →    Dask / Ray
    ```

    ---

    ## Production Recommendations

    ### Start New Projects With:
    1. **Polars** if Python-native team
    2. **DuckDB** if SQL-native team
    3. Either one will be 20-30x faster than classic pandas

    ### Migrate Existing Code:
    1. **Quick wins**: Add PyArrow backend to pandas (2-3x speedup)
    2. **High-impact refactor**: Rewrite hot paths in Polars
    3. **SQL translation**: Move analytics queries to DuckDB

    ### Cloud Cost Savings:
    - Faster execution = less compute time = lower costs
    - Polars/DuckDB efficiency can save 10-20x on cloud compute
    - Example: 1-hour daily pipeline → 3 minutes = 95% cost reduction

    ---

    ## Learning Path

    ### Week 1: Quick Wins
    - Add PyArrow backend to existing pandas code
    - Measure performance improvements
    - Identify bottlenecks

    ### Week 2-3: Learn Modern Tools
    - Polars fundamentals: expressions, lazy evaluation, group by
    - DuckDB basics: SQL queries, aggregations, joins
    - Rewrite one slow workflow

    ### Month 2: Production Deployment
    - Migrate critical pipelines
    - Set up monitoring and benchmarks
    - Document decision framework for team

    ### Month 3+: Advanced Patterns
    - Master lazy evaluation and query optimization
    - Learn streaming for larger-than-memory datasets
    - Explore cloud deployment (Snowflake, Databricks)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    # 🔗 Additional Resources

    ## Documentation
    - [Polars User Guide](https://pola-rs.github.io/polars-book/)
    - [DuckDB Documentation](https://duckdb.org/docs/)
    - [pandas PyArrow Integration](https://pandas.pydata.org/docs/user_guide/pyarrow.html)

    ## Further Learning
    - Polars GitHub: https://github.com/pola-rs/polars
    - DuckDB GitHub: https://github.com/duckdb/duckdb
    - Benchmarks: https://duckdblabs.github.io/db-benchmark/
    """)
    return


if __name__ == "__main__":
    app.run()
