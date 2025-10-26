import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # SQL for Data People

    This notebook provides hands-on, executable examples of advanced PostgreSQL concepts using real transaction and NYC taxi data.

    **📚 Reference:** This notebook follows marimo's SQL integration pattern. For more details, see the [DuckDB marimo guide](https://duckdb.org/docs/stable/guides/python/marimo).

    ## Learning Path
    1. **Setup and Data Loading** - Initialize environment and load datasets
    2. **Window Functions** - Ranking, aggregation, and analytical functions
    3. **Advanced PostgreSQL Commands** - QUALIFY, LATERAL, FILTER, and more
    4. **CTE vs Subquery Patterns** - Performance and readability comparisons
    5. **PIVOT/UNPIVOT Operations** - Data transformation techniques
    6. **Advanced SQL Patterns** - Gap analysis, deduplication, and statistical functions
    7. **Idempotent Operations** - UPSERT patterns and data synchronization
    8. **Performance Optimization** - Query analysis and debugging techniques

    Navigate through the cells sequentially for the best learning experience!
    """
    )
    return


@app.cell
def _():
    """
    ## Environment Setup and Imports

    Setting up all required libraries for our SQL learning environment.
    """
    import marimo as mo
    import polars as pl
    import duckdb
    import pyarrow.parquet as pq
    import warnings

    warnings.filterwarnings("ignore")

    mo.md("✅ **Environment Setup Complete**")
    return duckdb, mo, pl, pq


@app.cell
def _(duckdb, mo):
    """
    ## Database Setup with DuckDB

    Creating an in-memory DuckDB connection for SQL execution.
    No external database required!

    marimo will automatically discover this connection and use it with mo.sql()
    """
    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")

    # Test connection
    test_result = conn.execute("SELECT 'DuckDB connection successful!' as status").df()

    mo.md(f"""
    **Database Connection Status:** {test_result.loc[0, "status"]}

    🎯 **Why DuckDB?**
    - In-memory processing (no setup required)
    - Full SQL support including window functions
    - Excellent performance for analytical queries
    - More on this in a future episode!

    💡 **Using `mo.sql()`:**
    - marimo automatically discovers the `conn` variable
    - Use `mo.sql(f"...")` to execute queries with f-string interpolation
    - Results have a `.value` attribute containing the dataframe
    """)
    return (conn,)


@app.cell
def _(mo, pl):
    """
    ## Data Loading - Transaction Data

    Loading the synthetic transaction data with customer purchases, regions, and promo codes.
    """
    # Load transaction data from CSV
    transactions = pl.read_csv("season-1/data/transactions_synthetic.csv")

    # Convert date column to proper date type
    transactions = transactions.with_columns(pl.col("date").str.to_date().alias("date"))

    mo.md(f"""
    **📊 Transaction Data Loaded Successfully!**

    **Dataset Overview:**
    - **Records:** {len(transactions):,}
    - **Date Range:** {transactions["date"].min()} to {transactions["date"].max()}
    - **Customers:** {transactions["customer_id"].n_unique():,} unique
    - **Products:** {transactions["product_id"].n_unique():,} unique
    - **Regions:** {", ".join(sorted(transactions["region"].unique().to_list()))}
    """)
    return (transactions,)


@app.cell
def _(mo):
    mo.md("""**Schema:**""")
    return


@app.cell
def _(mo, transactions):
    """Display transaction schema"""
    _transactions_schema = mo.sql(f"DESCRIBE transactions")
    return


@app.cell
def _(mo):
    mo.md("""**Sample Data:**""")
    return


@app.cell
def _(mo, transactions):
    """Display sample transaction data"""
    _transactions_sample = mo.sql(
        f"""
        SELECT * FROM transactions 
        ORDER BY date 
        LIMIT 10
        """
    )
    return


@app.cell
def _(mo, pq):
    """
    ## Data Loading - NYC Taxi Data

    Loading NYC Yellow Taxi trip data for January and February 2024.
    """
    # Load taxi data from parquet files
    taxi_jan = pq.read_table("season-1/data/tlc/yellow_tripdata_2024-01.parquet")
    taxi_feb = pq.read_table("season-1/data/tlc/yellow_tripdata_2024-02.parquet")

    mo.md("""
    **🚕 NYC Taxi Data Loaded Successfully!**

    Two months of NYC Yellow Taxi data loaded (January and February 2024)
    """)
    return taxi_feb, taxi_jan


@app.cell
def _(mo):
    mo.md("""**Dataset Overview:**""")
    return


@app.cell
def _(mo, taxi_feb, taxi_jan):
    """Unified taxi data summary statistics"""
    taxi_summary_result = mo.sql(
        f"""
        WITH taxi_data AS (
            SELECT * FROM taxi_jan
            UNION ALL
            SELECT * FROM taxi_feb
        )
        SELECT 
            COUNT(*) as total_trips,
            MIN(DATE(tpep_pickup_datetime)) as earliest_date,
            MAX(DATE(tpep_pickup_datetime)) as latest_date,
            ROUND(AVG(fare_amount), 2) as avg_fare,
            ROUND(AVG(trip_distance), 2) as avg_distance,
            COUNT(DISTINCT DATE(tpep_pickup_datetime)) as unique_days
        FROM taxi_data
        """
    )
    return (taxi_summary_result,)


@app.cell
def _(mo):
    """
    # Window Functions - Basic Examples

    Demonstrating fundamental window function concepts with real transaction data.
    """
    mo.md("""
    ## 🪟 Window Functions - Rankings and Row Numbers

    Window functions allow us to perform calculations across related rows without collapsing them into groups.
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### ROW_NUMBER() - Sequential Numbering

    **ROW_NUMBER()** assigns unique sequential integers to rows within partitions:
    """
    )
    return


@app.cell
def _(mo, transactions):
    """ROW_NUMBER window function example"""
    _row_number_result = mo.sql(
        f"""
        SELECT 
            date,
            customer_id,
            region,
            price * quantity as transaction_amount,
            ROW_NUMBER() OVER (PARTITION BY region ORDER BY date) as row_num_by_region,
            ROW_NUMBER() OVER (ORDER BY price * quantity DESC) as row_num_by_amount
        FROM transactions
        WHERE date >= '2024-01-01' AND date <= '2024-01-31'  -- January only
        ORDER BY region, date
        LIMIT 15
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    💡 **Key Insights:**
    - `row_num_by_region`: Sequential numbering within each region by date
    - `row_num_by_amount`: Overall ranking by transaction amount (highest first)
    - Notice how each row gets a unique number even for tied values
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### RANK() vs DENSE_RANK() - Handling Ties""")
    return


@app.cell
def _(mo, transactions):
    """Ranking functions comparison"""
    _ranking_result = mo.sql(
        f"""
        WITH customer_spending AS (
            SELECT 
                customer_id,
                SUM(price * quantity) as total_spent,
                COUNT(*) as transaction_count
            FROM transactions
            GROUP BY customer_id
        )
        SELECT 
            customer_id,
            total_spent,
            transaction_count,
            RANK() OVER (ORDER BY transaction_count DESC) as rank_with_gaps,
            DENSE_RANK() OVER (ORDER BY transaction_count DESC) as dense_rank,
            NTILE(5) OVER (ORDER BY transaction_count DESC) as quintile
        FROM customer_spending
        ORDER BY transaction_count DESC, customer_id
        LIMIT 20
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    🎯 **Understanding the Differences:**
    - **ROW_NUMBER()**: Always increments sequentially (1, 2, 3, 4, 5...) even with ties - each row gets a unique number
    - **RANK()**: Leaves gaps after tied values (1, 2, 2, 4, 5...)
    - **DENSE_RANK()**: No gaps after tied values (1, 2, 2, 3, 4...)
    - **NTILE(5)**: Divides data into 5 equal-sized groups (quintiles)
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Window Aggregations - Running Totals and Moving Averages""")
    return


@app.cell
def _(mo, transactions):
    """Running totals and moving averages"""
    window_agg_result = mo.sql(
        f"""
        WITH daily_sales AS (
            SELECT 
                date,
                SUM(price * quantity) as daily_revenue,
                COUNT(*) as daily_transactions,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM transactions
            WHERE date BETWEEN '2024-01-01' AND '2024-02-28'  -- Jan-Feb
            GROUP BY date
            ORDER BY date
        )
        SELECT 
            date,
            daily_revenue,
            daily_transactions,
            unique_customers,
            -- Running totals
            SUM(daily_revenue) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) as cumulative_revenue,
            SUM(daily_transactions) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) as cumulative_transactions,

            -- Moving averages (7-day window)
            ROUND(
                AVG(daily_revenue) OVER (
                    ORDER BY date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 2
            ) as seven_day_avg_revenue,

            ROUND(
                AVG(daily_transactions) OVER (
                    ORDER BY date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 2
            ) as seven_day_avg_transactions
        FROM daily_sales
        ORDER BY date
        """
    )
    return (window_agg_result,)


@app.cell
def _(mo, window_agg_result):
    mo.md(
        f"""
    **📈 Running Calculations and Moving Averages Results:**

    **Dataset Overview:**
        - **Total Days:** {window_agg_result.height}
        - **Total Revenue:** ${window_agg_result.select("cumulative_revenue").tail(1).item():,.2f}
        - **Average Daily Revenue:** ${window_agg_result["daily_revenue"].mean():,.2f}

    🔍 **Window Frame Explanation:**
    - `ROWS UNBOUNDED PRECEDING`: From start to current row (running total)
    - `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`: 7-day moving window
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### LAG() and LEAD() - Accessing Previous/Next Values""")
    return


@app.cell
def _(mo, transactions):
    """Time series analysis with LAG and LEAD"""
    _lag_lead_result = mo.sql(
        f"""
        WITH monthly_metrics AS (
            SELECT 
                DATE_TRUNC('month', date) as month,
                SUM(price * quantity) as monthly_revenue,
                COUNT(*) as monthly_transactions,
                COUNT(DISTINCT customer_id) as unique_customers_per_month
            FROM transactions
            GROUP BY DATE_TRUNC('month', date)
            ORDER BY month
        )
        SELECT 
            month,
            monthly_revenue,
            monthly_transactions,
            unique_customers_per_month,

            -- Previous month comparisons
            LAG(monthly_revenue, 1) OVER (ORDER BY month) as prev_month_revenue,
            monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY month) as revenue_change,

            ROUND(
                (monthly_revenue - LAG(monthly_revenue, 1) OVER (ORDER BY month)) / 
                NULLIF(LAG(monthly_revenue, 1) OVER (ORDER BY month), 0) * 100, 
                2
            ) as revenue_change_pct,

            -- Next month preview
            LEAD(monthly_revenue, 1) OVER (ORDER BY month) as next_month_revenue,

            -- First and last values in the dataset
            FIRST_VALUE(monthly_revenue) OVER (ORDER BY month ROWS UNBOUNDED PRECEDING) as first_month_revenue,
            LAST_VALUE(monthly_revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_month_revenue

        FROM monthly_metrics
        ORDER BY month
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    💡 **Advanced Window Functions:**
    - **LAG(column, 1)**: Previous row value (great for month-over-month analysis)
    - **LEAD(column, 1)**: Next row value (forecasting context)
    - **FIRST_VALUE()**: First value in the window (baseline comparison)
    - **LAST_VALUE()**: Last value in the window (endpoint comparison)

    🎯 **Business Insights:**
    - Track month-over-month growth rates
    - Compare current performance to baseline (first month)
    - Identify trends and seasonality patterns
    """
    )
    return


@app.cell
def _(mo):
    """
    # Advanced PostgreSQL Commands

    Exploring powerful PostgreSQL features for complex analytical scenarios.
    """
    mo.md("""
    ## 🚀 Advanced PostgreSQL Commands

    Moving beyond basic SQL to leverage PostgreSQL's advanced analytical capabilities.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### QUALIFY Clause - Filtering Window Function Results""")
    return


@app.cell
def _(mo, transactions):
    """Top 3 customers by spending in each region using QUALIFY"""
    _qualify_result = mo.sql(
        f"""
        WITH customer_regional_spending AS (
            SELECT 
                customer_id,
                region,
                SUM(price * quantity) as total_spent,
                COUNT(*) as transaction_count,
                ROUND(AVG(price * quantity), 2) as avg_transaction_amount
            FROM transactions
            GROUP BY customer_id, region
        )
        SELECT 
            customer_id,
            region,
            total_spent,
            transaction_count,
            avg_transaction_amount,
            RANK() OVER (PARTITION BY region ORDER BY total_spent DESC) as region_rank
        FROM customer_regional_spending
        QUALIFY RANK() OVER (PARTITION BY region ORDER BY total_spent DESC) <= 3
        ORDER BY region, total_spent DESC
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Why QUALIFY is Powerful:**
    - ✅ Can filter directly on window function results
    - ✅ More readable than subqueries for this use case
    - ✅ Executes after window functions are computed

    **Traditional Alternative (without QUALIFY):**
    ```sql
    SELECT * FROM (
        SELECT *, RANK() OVER (...) as rank
        FROM table
    ) ranked
    WHERE rank <= 3;
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### FILTER Clause - Conditional Aggregations""")
    return


@app.cell
def _(mo, transactions):
    """Analyze promo code effectiveness by region"""
    _filter_clause_result = mo.sql(
        f"""
        SELECT 
            region,
            COUNT(*) as total_transactions,

            -- Count transactions by promo code
            COUNT(*) FILTER (WHERE promo_code = 'PROMO10') as promo10_count,
            COUNT(*) FILTER (WHERE promo_code = 'PROMO20') as promo20_count,
            COUNT(*) FILTER (WHERE promo_code = 'PROMO30') as promo30_count,
            COUNT(*) FILTER (WHERE promo_code IS NULL) as no_promo_count,

            -- Average prices by promo type
            ROUND(AVG(price) FILTER (WHERE promo_code = 'PROMO10'), 2) as avg_price_promo10,
            ROUND(AVG(price) FILTER (WHERE promo_code = 'PROMO20'), 2) as avg_price_promo20,
            ROUND(AVG(price) FILTER (WHERE promo_code = 'PROMO30'), 2) as avg_price_promo30,
            ROUND(AVG(price) FILTER (WHERE promo_code IS NULL), 2) as avg_price_no_promo,

            -- Total revenue by promo type
            ROUND(SUM(price * quantity) FILTER (WHERE promo_code = 'PROMO10'), 2) as revenue_promo10,
            ROUND(SUM(price * quantity) FILTER (WHERE promo_code = 'PROMO20'), 2) as revenue_promo20,
            ROUND(SUM(price * quantity) FILTER (WHERE promo_code = 'PROMO30'), 2) as revenue_promo30

        FROM transactions
        GROUP BY region
        ORDER BY total_transactions DESC
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **💡 FILTER Clause Benefits:**
    - **Single Query**: Calculate multiple conditional metrics in one pass
    - **Performance**: More efficient than multiple subqueries
    - **Readability**: Clear intent with explicit conditions

    **Business Insights:**
    - Compare promo code effectiveness across regions
    - Identify which promotions drive higher average prices
    - Analyze revenue distribution by promotion type
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Statistical Functions with WITHIN GROUP""")
    return


@app.cell
def _(mo, transactions):
    """Comprehensive statistical analysis by region"""
    _stats_result = mo.sql(
        f"""
        WITH transaction_amounts AS (
            SELECT 
                region,
                price * quantity as amount
            FROM transactions
            WHERE date >= '2024-01-01'
        )
        SELECT 
            region,
            COUNT(*) as transaction_count,

            -- Central tendency
            ROUND(AVG(amount), 2) as mean_amount,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount), 2) as median_amount,

            -- Quartiles
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount), 2) as q1_amount,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount), 2) as q3_amount,

            -- Spread measures  
            ROUND(STDDEV(amount), 2) as std_deviation,
            ROUND(MAX(amount) - MIN(amount), 2) as range_amount,

            -- Extreme values
            ROUND(MIN(amount), 2) as min_amount,
            ROUND(MAX(amount), 2) as max_amount,
            ROUND(PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY amount), 2) as p1_amount,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount), 2) as p99_amount

        FROM transaction_amounts
        GROUP BY region
        ORDER BY median_amount DESC
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🎯 Statistical Functions Explained:**
    - **PERCENTILE_CONT(0.5)**: Continuous percentile (median)
    - **WITHIN GROUP (ORDER BY ...)**: Specifies ordering for percentile calculations
    - **Quartiles (Q1, Q3)**: 25th and 75th percentiles for outlier detection
    - **P1, P99**: 1st and 99th percentiles to identify extreme values

    **Business Applications:**
    - **Outlier Detection**: Use IQR (Q3 - Q1) * 1.5 rule
    - **Regional Comparison**: Compare distributions across regions
    - **Pricing Strategy**: Understand transaction amount patterns
    """
    )
    return


@app.cell
def _(mo):
    """
    # CTE vs Subquery Performance Comparison

    Demonstrating when to use CTEs versus subqueries with practical examples.
    """
    mo.md("""
    ## 🔄 CTE vs Subquery Decision Framework

    Understanding when to use Common Table Expressions versus subqueries for optimal performance and readability.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### CTE Approach - Complex Multi-Step Analysis""")
    return


@app.cell
def _(mo, transactions):
    """CTE approach: Clear, readable multi-step analysis"""
    _cte_result = mo.sql(
        f"""
        WITH customer_metrics AS (
            -- Step 1: Calculate basic customer metrics
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent,
                ROUND(AVG(price * quantity), 2) as avg_transaction_value,
                MIN(date) as first_purchase_date,
                MAX(date) as last_purchase_date
            FROM transactions
            GROUP BY customer_id
        ),
        customer_segments AS (
            -- Step 2: Segment customers based on spending and frequency
            SELECT 
                *,
                CASE 
                    WHEN total_spent >= 1000 AND transaction_count >= 10 THEN 'High Value High Frequency'
                    WHEN total_spent >= 1000 THEN 'High Value Low Frequency' 
                    WHEN transaction_count >= 10 THEN 'Low Value High Frequency'
                    ELSE 'Low Value Low Frequency'
                END as customer_segment,
                DATEDIFF('day', first_purchase_date, last_purchase_date) as customer_lifespan_days
            FROM customer_metrics
        ),
        segment_analysis AS (
            -- Step 3: Analyze segments
            SELECT 
                customer_segment,
                COUNT(*) as count_customers,
                ROUND(AVG(total_spent), 2) as avg_spend_per_customer,
                ROUND(AVG(transaction_count), 2) as avg_transactions_per_customer,
                ROUND(AVG(customer_lifespan_days), 1) as avg_customer_lifespan_days,
                SUM(total_spent) as total_segment_revenue
            FROM customer_segments
            GROUP BY customer_segment
        )
        -- Step 4: Final output with business context
        SELECT 
            customer_segment,
            count_customers,
            avg_spend_per_customer,
            avg_transactions_per_customer, 
            avg_customer_lifespan_days,
            total_segment_revenue,
            ROUND(total_segment_revenue / SUM(total_segment_revenue) OVER () * 100, 1) as revenue_percentage
        FROM segment_analysis
        ORDER BY total_segment_revenue DESC
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🎯 When CTEs Excel:**
    - **Complex Logic**: Multiple transformation steps
    - **Readability**: Clear, logical flow from raw data to insights
    - **Reusability**: Same CTE referenced multiple times
    - **Maintainability**: Easy to modify individual steps
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Subquery Approach - Simple Filtering and Comparisons""")
    return


@app.cell
def _(mo, transactions):
    """Subquery approach: Efficient for simple operations"""
    _subquery_result = mo.sql(
        f"""
        SELECT 
            t.customer_id,
            t.date,
            t.price * t.quantity as transaction_amount,
            t.region,
            -- Compare to customer average (correlated subquery)
            ROUND(
                (t.price * t.quantity) - 
                (SELECT AVG(t2.price * t2.quantity) 
                 FROM transactions t2 
                 WHERE t2.customer_id = t.customer_id), 
                2
            ) as diff_from_customer_avg,

            -- Compare to regional average (non-correlated subquery)
            ROUND(
                (t.price * t.quantity) - 
                (SELECT AVG(t3.price * t3.quantity) 
                 FROM transactions t3 
                 WHERE t3.region = t.region), 
                2
            ) as diff_from_regional_avg,

            -- Count of customer's previous transactions
            (SELECT COUNT(*) 
             FROM transactions t4 
             WHERE t4.customer_id = t.customer_id 
             AND t4.date < t.date) as previous_transaction_count

        FROM transactions t
        WHERE t.customer_id IN (
            -- Subquery for filtering: customers with high spending
            SELECT customer_id
            FROM transactions
            GROUP BY customer_id
            HAVING SUM(price * quantity) > 500
        )
        AND t.date >= '2024-01-01'
        ORDER BY t.customer_id, t.date
        LIMIT 20
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🎯 When Subqueries Excel:**
    - **Simple Filtering**: WHERE clause with EXISTS or IN
    - **Correlated Operations**: Row-by-row comparisons
    - **Performance**: Often optimized better by query planner
    - **Memory Efficiency**: No intermediate result materialization
    """
    )
    return


@app.cell
def _(mo):
    """
    ### Performance Comparison Analysis
    """
    performance_comparison = """
    We can analyze both approaches to understand their trade-offs:

    **CTE Advantages:**
    - **Readability**: Complex logic broken into clear steps
    - **Maintainability**: Easy to modify individual components
    - **Debugging**: Can inspect intermediate results
    - **Reusability**: Same CTE used multiple times in one query

    **Subquery Advantages:**
    - **Performance**: Often optimized better by query planner
    - **Memory**: No intermediate materialization for simple operations
    - **Simplicity**: Direct approach for straightforward operations
    - **Flexibility**: Can be correlated or non-correlated as needed
    """

    mo.md(f"""
    **⚖️ Performance and Readability Trade-offs:**

    {performance_comparison}

    **🎯 Decision Framework:**

    | Scenario | Recommendation | Reason |
    |----------|----------------|---------|
    | Multi-step transformations | CTE | Better readability |
    | Simple filtering | Subquery | Better performance |
    | Reusing same logic | CTE | Avoid duplication |
    | Row-by-row comparisons | Subquery | Natural fit |
    | Complex business logic | CTE | Easier maintenance |
    | Performance-critical | Test both | Results vary |
    """)
    return


@app.cell
def _(mo):
    """
    ### Simple CTE and Subquery Examples with Execution Plans
    """
    mo.md("""
    ## 🔍 Simple CTE vs Subquery with Execution Plans

    Let's create simpler examples to clearly demonstrate the differences between CTEs and subqueries, 
    then analyze their execution plans to understand performance characteristics.

    **What We'll Compare:**
    - Simple customer analysis using CTE approach
    - Equivalent analysis using subquery approach  
    - Execution plans for both approaches
    - Performance timing comparison
    """)
    return


@app.cell
def _(mo):
    mo.md("""### Simple CTE Example - Customer Analysis""")
    return


@app.cell
def _(mo, transactions):
    """Simple CTE approach: Find top customers by spending"""
    _simple_cte_result = mo.sql(
        f"""
        WITH customer_totals AS (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
        ),
        top_customers AS (
            SELECT 
                customer_id,
                transaction_count,
                total_spent,
                ROUND(total_spent / transaction_count, 2) as avg_per_transaction
            FROM customer_totals
            WHERE total_spent > 300
        )
        SELECT 
            customer_id,
            transaction_count,
            total_spent,
            avg_per_transaction
        FROM top_customers
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Simple Subquery Example - Equivalent Analysis""")
    return


@app.cell
def _(mo, transactions):
    """Simple subquery approach: Equivalent analysis"""
    _simple_subquery_result = mo.sql(
        f"""
        SELECT 
            customer_id,
            transaction_count,
            total_spent,
            ROUND(total_spent / transaction_count, 2) as avg_per_transaction
        FROM (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
            HAVING SUM(price * quantity) > 300
        ) as customer_totals
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )
    return


@app.cell
def _():
    """Query plan formatting functions"""

    def format_query_plan_tree(explain_result):
        """Format DuckDB EXPLAIN output preserving tree structure"""
        if explain_result.height == 0:
            return "No query plan available"

        # Get the physical plan from the result
        plan_text = explain_result.select("explain_value").item()

        # Clean up the ASCII art while preserving structure
        lines = plan_text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Replace Unicode box drawing characters with simpler ASCII
            cleaned_line = (
                line.replace("┌", "+")
                .replace("┐", "+")
                .replace("└", "+")
                .replace("┘", "+")
            )
            cleaned_line = (
                cleaned_line.replace("─", "-")
                .replace("│", "|")
                .replace("┬", "+")
                .replace("┴", "+")
            )
            cleaned_line = cleaned_line.replace("├", "+").replace("┤", "+")

            # Clean up extra spaces and make it more readable
            cleaned_line = cleaned_line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines)

    def format_query_plan(explain_result):
        """Format DuckDB EXPLAIN output for better readability"""
        if explain_result.height == 0:
            return "No query plan available"

        # Get the physical plan from the result
        plan_text = explain_result.select("explain_value").item()

        # Parse the ASCII art plan into a more readable format
        lines = plan_text.split("\n")
        formatted_lines = []

        # Track the current operation being processed
        current_operation = None
        operation_details = []

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # Check if this is a new operation box
            if "┌" in clean_line and "┐" in clean_line:
                # Save previous operation if exists
                if current_operation:
                    formatted_lines.append(f"**{current_operation}**")
                    for detail in operation_details:
                        formatted_lines.append(f"  • {detail}")
                    formatted_lines.append("")  # Add spacing
                    operation_details = []

                # Extract operation name
                operation = (
                    clean_line.replace("┌", "")
                    .replace("┐", "")
                    .replace("─", "")
                    .strip()
                )
                current_operation = operation

            # Check if this is content within an operation box
            elif "│" in clean_line and not clean_line.startswith("│"):
                content = clean_line.replace("│", "").strip()
                if content and not content.startswith("─") and content != "":
                    operation_details.append(content)
            elif clean_line.startswith("│") and "│" in clean_line[1:]:
                content = clean_line.replace("│", "").strip()
                if content and not content.startswith("─") and content != "":
                    operation_details.append(content)

        # Don't forget the last operation
        if current_operation:
            formatted_lines.append(f"**{current_operation}**")
            for detail in operation_details:
                formatted_lines.append(f"  • {detail}")

        return "\n".join(formatted_lines)
    return format_query_plan, format_query_plan_tree


@app.cell
def _(mo):
    mo.md("""### Execution Plan Comparison""")
    return


@app.cell
def _(format_query_plan, format_query_plan_tree, mo):
    """EXPLAIN for simple CTE query"""
    _explain_simple_cte = mo.sql(
        f"""
        EXPLAIN
        WITH customer_totals AS (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
        ),
        top_customers AS (
            SELECT 
                customer_id,
                transaction_count,
                total_spent,
                ROUND(total_spent / transaction_count, 2) as avg_per_transaction
            FROM customer_totals
            WHERE total_spent > 300
        )
        SELECT 
            customer_id,
            transaction_count,
            total_spent,
            avg_per_transaction
        FROM top_customers
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )

    mo.md(f"""
    **🔍 CTE Execution Plan:**

    **Structured View:**
    ```
    {format_query_plan(_explain_simple_cte)}
    ```

    **Tree Structure View:**
    ```
    {format_query_plan_tree(_explain_simple_cte)}
    ```
    """)
    return


@app.cell
def _(format_query_plan, format_query_plan_tree, mo):
    """EXPLAIN for simple subquery"""
    _explain_simple_subquery = mo.sql(
        f"""
        EXPLAIN
        SELECT 
            customer_id,
            transaction_count,
            total_spent,
            ROUND(total_spent / transaction_count, 2) as avg_per_transaction
        FROM (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
            HAVING SUM(price * quantity) > 300
        ) as customer_totals
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )

    mo.md(f"""
    **🔍 Subquery Execution Plan:**

    **Structured View:**
    ```
    {format_query_plan(_explain_simple_subquery)}
    ```

    **Tree Structure View:**
    ```
    {format_query_plan_tree(_explain_simple_subquery)}
    ```
    """)
    return


@app.cell
def _(mo):
    """
    ### Query Plans - Understanding EXPLAIN
    """
    mo.md("""
    ## 🎯 Query Plan Analysis: What's Optimal?

    ### Understanding the Plans

    **For Simple Queries (like our examples):**
    - Both CTE and subquery approaches often produce **identical or very similar plans**
    - Modern query optimizers (DuckDB, PostgreSQL) are smart enough to recognize equivalent logic
    - The optimizer may "flatten" CTEs into subqueries internally for better performance

    **Optimal Plan Characteristics:**

    1. **Efficient Scan Methods**
       - ✅ **Index Scan**: Fast when filtering on indexed columns
       - ⚠️ **Sequential Scan**: Acceptable for small-medium tables or when filtering many rows
       - ❌ Avoid full table scans on large tables with selective predicates

    2. **Smart Aggregation**
       - ✅ **Hash Aggregate**: Fast for most GROUP BY operations
       - ✅ **Partial Aggregation**: Parallel processing when available
       - The `HAVING` clause in subqueries can be more efficient by filtering during aggregation

    3. **Minimal Materialization**
       - ✅ Subqueries often avoid intermediate result materialization
       - ⚠️ CTEs may materialize results (though modern optimizers optimize this)
       - DuckDB and PostgreSQL 12+ often inline CTEs for better performance

    ### When Plans Diverge

    **CTEs May Be Less Optimal When:**
    - Referenced multiple times → May materialize once and reuse (can be good or bad)
    - Optimizer can't inline them → Creates temporary result sets
    - Contains recursive logic → Different execution path

    **Subqueries May Be Less Optimal When:**
    - Correlated subqueries → Executed once per outer row
    - Complex nested subqueries → Harder for optimizer to rewrite
    - Multiple references → Code duplication leads to redundant computation

    ### Real-World Optimization Tips

    1. **Start with Readability**: Write CTEs for clarity, especially for complex logic
    2. **Measure Performance**: Use `EXPLAIN ANALYZE` with actual data volumes
    3. **Consider Materialized CTEs**: PostgreSQL 12+ allows `WITH ... AS MATERIALIZED`
    4. **Watch for Correlation**: Correlated subqueries can be performance killers
    5. **Index Appropriately**: The best query plan depends on having the right indexes

    **The Bottom Line:**
    For most analytical queries, **readability and maintainability matter more than micro-optimizations**. 
    The difference between CTE and subquery is often negligible with modern optimizers. 
    Focus on clear logic first, then optimize based on actual performance data.
    """)
    return


@app.cell
def _(mo):
    """
    # PIVOT and UNPIVOT Operations

    Demonstrating data transformation techniques in PostgreSQL/DuckDB.
    """
    mo.md("""
    ## 🔄 PIVOT and UNPIVOT Operations

    PostgreSQL doesn't have native PIVOT/UNPIVOT operators, but we can achieve the same results using various techniques.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### Manual Pivot - Sales by Region and Month""")
    return


@app.cell
def _(mo, transactions):
    """Transform region data from rows to columns by month"""
    _pivot_result = mo.sql(
        f"""
        WITH monthly_regional_sales AS (
            SELECT 
                DATE_TRUNC('month', date) as month,
                region,
                SUM(price * quantity) as sales
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', date), region
        )
        SELECT 
            month,
            ROUND(COALESCE(SUM(CASE WHEN region = 'North' THEN sales END), 0), 2) as north_sales,
            ROUND(COALESCE(SUM(CASE WHEN region = 'South' THEN sales END), 0), 2) as south_sales,
            ROUND(COALESCE(SUM(CASE WHEN region = 'East' THEN sales END), 0), 2) as east_sales,
            ROUND(COALESCE(SUM(CASE WHEN region = 'West' THEN sales END), 0), 2) as west_sales,
            ROUND(SUM(sales), 2) as total_monthly_sales
        FROM monthly_regional_sales
        GROUP BY month
        ORDER BY month
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🔧 Pivot Technique Explained:**
    - **CASE WHEN**: Conditionally aggregate values for each region
    - **COALESCE**: Handle NULL values (months with no sales in a region)
    - **GROUP BY month**: Collapse regions into columns for each month

    **Use Cases:**
    - **Reporting**: Month-over-month regional performance
    - **Dashboards**: Wide format for visualization tools
    - **Analysis**: Easy comparison across categories
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Advanced Pivot - Customer Segments by Promo Code Usage""")
    return


@app.cell
def _(mo, transactions):
    """Analyze customer segments and their promo code preferences"""
    _advanced_pivot_result = mo.sql(
        f"""
        WITH customer_promo_analysis AS (
            SELECT 
                customer_id,
                -- Customer spending tier
                CASE 
                    WHEN SUM(price * quantity) >= 1000 THEN 'High Spender'
                    WHEN SUM(price * quantity) >= 500 THEN 'Medium Spender' 
                    ELSE 'Low Spender'
                END as spending_tier,
                -- Promo code usage counts
                COUNT(CASE WHEN promo_code = 'PROMO10' THEN 1 END) as promo10_usage,
                COUNT(CASE WHEN promo_code = 'PROMO20' THEN 1 END) as promo20_usage,
                COUNT(CASE WHEN promo_code = 'PROMO30' THEN 1 END) as promo30_usage,
                COUNT(CASE WHEN promo_code IS NULL THEN 1 END) as no_promo_usage,
                COUNT(*) as total_transactions
            FROM transactions
            GROUP BY customer_id
        )
        SELECT 
            spending_tier,
            COUNT(*) as customers_in_tier,

            -- Average promo usage per customer in each tier
            ROUND(AVG(promo10_usage), 2) as avg_promo10_per_customer,
            ROUND(AVG(promo20_usage), 2) as avg_promo20_per_customer,
            ROUND(AVG(promo30_usage), 2) as avg_promo30_per_customer,
            ROUND(AVG(no_promo_usage), 2) as avg_no_promo_per_customer,

            -- Total promo usage by tier
            SUM(promo10_usage) as total_promo10_usage,
            SUM(promo20_usage) as total_promo20_usage,
            SUM(promo30_usage) as total_promo30_usage,
            SUM(no_promo_usage) as total_no_promo_usage,

            -- Promo preference percentages
            ROUND(SUM(promo10_usage) * 100.0 / SUM(total_transactions), 1) as promo10_pct,
            ROUND(SUM(promo20_usage) * 100.0 / SUM(total_transactions), 1) as promo20_pct,
            ROUND(SUM(promo30_usage) * 100.0 / SUM(total_transactions), 1) as promo30_pct

        FROM customer_promo_analysis
        GROUP BY spending_tier
        ORDER BY 
            CASE spending_tier
                WHEN 'High Spender' THEN 1
                WHEN 'Medium Spender' THEN 2
                ELSE 3
            END
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **💡 Business Insights:**
    - **Promo Effectiveness**: Which promotions appeal to each spending tier?
    - **Customer Behavior**: Do high spenders use promotions differently?
    - **Marketing Strategy**: Target promotions based on customer segments

    **Cross-Tabulation Benefits:**
    - **Multi-dimensional Analysis**: Customer tier × Promo code usage
    - **Percentage Calculations**: Understanding proportional preferences
    - **Segmentation Insights**: Behavioral patterns by customer value
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### UNPIVOT - Converting Wide to Long Format""")
    return


@app.cell
def _(mo, transactions):
    """Unpivot: Convert wide to long format"""
    _unpivot_result = mo.sql(
        f"""
        WITH pivoted_data AS (
            SELECT 
                DATE_TRUNC('month', date) as month,
                ROUND(SUM(CASE WHEN region = 'North' THEN price * quantity ELSE 0 END), 2) as north_sales,
                ROUND(SUM(CASE WHEN region = 'South' THEN price * quantity ELSE 0 END), 2) as south_sales,
                ROUND(SUM(CASE WHEN region = 'East' THEN price * quantity ELSE 0 END), 2) as east_sales,
                ROUND(SUM(CASE WHEN region = 'West' THEN price * quantity ELSE 0 END), 2) as west_sales
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', date)
        )
        -- Unpivot back to normalized form using UNION ALL
        SELECT month, 'North' as region, north_sales as sales FROM pivoted_data
        UNION ALL
        SELECT month, 'South' as region, south_sales as sales FROM pivoted_data  
        UNION ALL
        SELECT month, 'East' as region, east_sales as sales FROM pivoted_data
        UNION ALL
        SELECT month, 'West' as region, west_sales as sales FROM pivoted_data
        ORDER BY month, region
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🛠️ UNPIVOT Techniques:**
    - **UNION ALL**: Most common method for unpivoting in PostgreSQL
    - **VALUES Clause**: Alternative approach for smaller datasets
    - **LATERAL Joins**: More complex but flexible unpivoting

    **When to UNPIVOT:**
    - **Data Integration**: Combining data from different sources
    - **Visualization**: Many tools prefer long format
    - **Analysis**: Easier to filter and aggregate normalized data
    """
    )
    return


@app.cell
def _(mo):
    """
    # Idempotent Operations

    Building robust, repeatable data operations that can be run multiple times safely.
    """
    mo.md("""
    ## ♻️ Idempotent Operations

    Creating data operations that produce the same result regardless of how many times they're executed.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### Setting Up Tables for UPSERT Examples""")
    return


@app.cell
def _(conn, pl, transactions):
    """Create customer_summary DataFrame and register with DuckDB"""
    # Create customer summary data using Polars
    customer_summary = (
        transactions.group_by("customer_id")
        .agg(
            [
                pl.col("date").min().alias("first_purchase_date"),
                pl.col("date").max().alias("last_purchase_date"),
                pl.len().alias("total_transactions"),
                (pl.col("price") * pl.col("quantity")).sum().alias("total_spent"),
                (pl.col("price") * pl.col("quantity"))
                .mean()
                .alias("avg_transaction_amount"),
                pl.col("region").mode().first().alias("favorite_region"),
                pl.lit(None, dtype=pl.Datetime).alias("last_updated"),
            ]
        )
        .with_columns(
            pl.col("total_spent").round(2), pl.col("avg_transaction_amount").round(2)
        )
    )

    # Register the DataFrame with DuckDB for mo.sql() queries
    conn.register("customer_summary", customer_summary)
    return (customer_summary,)


@app.cell
def _(conn, pl, transactions):
    """Create product_metrics DataFrame and register with DuckDB"""
    # Create product metrics data using Polars
    product_metrics = (
        transactions.filter(pl.col("date") >= pl.date(2024, 1, 1))
        .group_by("product_id")
        .agg(
            [
                (pl.col("price") * pl.col("quantity")).sum().alias("total_revenue"),
                pl.col("quantity").sum().alias("total_quantity_sold"),
                pl.col("price").mean().alias("avg_price"),
                pl.col("customer_id").n_unique().alias("unique_customers"),
                pl.col("date").min().alias("first_sale_date"),
                pl.col("date").max().alias("last_sale_date"),
                pl.lit("2024-01-01").alias("calculation_date"),
            ]
        )
        .filter(
            pl.col("total_quantity_sold") >= 5  # Only products with 5+ sales
        )
        .with_columns([pl.col("total_revenue").round(2), pl.col("avg_price").round(2)])
    )

    # Register the DataFrame with DuckDB for mo.sql() queries
    conn.register("product_metrics", product_metrics)
    return (product_metrics,)


@app.cell
def _(mo):
    mo.md(
        """
    **🛠️ Created Tables for UPSERT Examples:**
    - `customer_summary`: Aggregated customer metrics
    - `product_metrics`: Product performance statistics

    These tables will demonstrate idempotent insert/update operations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Basic UPSERT Pattern - Customer Summary""")
    return


@app.cell
def _(customer_summary, mo):
    """Demonstrate UPSERT concept with Polars DataFrames"""
    mo.md(f"""
    **📊 Customer Summary Data Created Successfully!**

    **Dataset Overview:**
    - **Total Customers:** {len(customer_summary):,}
    - **Average Customer Value:** ${customer_summary["total_spent"].mean():,.2f}
    - **Average Transactions per Customer:** {customer_summary["total_transactions"].mean():.1f}

    **🎯 UPSERT Concept with Polars:**
    - **DataFrame Creation**: Equivalent to INSERT operations
    - **DataFrame Updates**: Use `.with_columns()` or `.join()` for updates
    - **Idempotent Operations**: Can recreate DataFrames safely
    - **Performance**: Polars operations are vectorized and fast
    """)
    return


@app.cell
def _(mo):
    mo.md("""**Summary Statistics:**""")
    return


@app.cell
def _(customer_summary, mo):
    """Check UPSERT results"""
    _upsert_summary_result = mo.sql(
        f"""
        SELECT 
            COUNT(*) as customers_processed,
            MIN(first_purchase_date) as earliest_first_purchase,
            MAX(last_purchase_date) as latest_last_purchase,
            ROUND(AVG(total_spent), 2) as avg_customer_value,
            ROUND(SUM(total_spent), 2) as total_customer_value
        FROM customer_summary
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""**Top Customers by Spending:**""")
    return


@app.cell
def _(customer_summary, mo):
    """Show sample records"""
    _sample_customers_result = mo.sql(
        f"""
        SELECT 
            customer_id,
            first_purchase_date,
            last_purchase_date,
            total_transactions,
            total_spent,
            avg_transaction_amount,
            favorite_region
        FROM customer_summary
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **🎯 UPSERT Benefits:**
    - **Idempotent**: Can run multiple times safely
    - **Efficient**: Updates only when conflicts occur
    - **Data Integrity**: Maintains referential consistency
    - **Flexible**: Handles both new and existing records
    """
    )
    return


@app.cell
def _(mo, product_metrics):
    """
    ### Advanced UPSERT with Conditional Logic
    """
    mo.md(f"""
    **📊 Advanced Product Metrics Analysis:**

    **Dataset Overview:**
    - **Products Analyzed:** {len(product_metrics):,}
    - **Total Product Revenue:** ${product_metrics["total_revenue"].sum():,.2f}
    - **Average Revenue per Product:** ${product_metrics["total_revenue"].mean():,.2f}
    - **Average Price:** ${product_metrics["avg_price"].mean():,.2f}

    **🚀 Advanced DataFrame Operations:**
    - **Conditional Filtering**: Only products with 5+ sales included
    - **Data Quality**: Filtered to 2024 data only
    - **Derived Metrics**: Revenue per customer calculations
    - **Business Logic**: Comprehensive product performance analysis

    **🎯 Polars DataFrame Benefits:**
    - **Vectorized Operations**: Fast processing of large datasets
    - **Lazy Evaluation**: Optimized query execution
    - **Type Safety**: Strong typing prevents data quality issues
    - **Memory Efficient**: Optimized memory usage patterns
    """)
    return


@app.cell
def _(conn, pl):
    """Create sync_metadata DataFrame and register with DuckDB"""
    # Create sync metadata DataFrame using Polars
    sync_metadata = pl.DataFrame(
        {
            "table_name": ["customer_summary"],
            "last_sync_timestamp": ["2024-01-01 00:00:00"],
            "records_processed": [0],
            "last_sync_date": ["2024-01-01"],
        }
    )

    # Register the DataFrame with DuckDB for mo.sql() queries
    conn.register("sync_metadata", sync_metadata)
    return (sync_metadata,)


@app.cell
def _(mo, sync_metadata):
    """
    ### Incremental Update Pattern with Metadata Tracking
    """
    sync_status_result = mo.sql(f"""
        SELECT 
            table_name,
            last_sync_timestamp,
            records_processed,
            last_sync_date,
            CURRENT_TIMESTAMP - CAST(last_sync_timestamp AS TIMESTAMP) as time_since_last_sync
        FROM sync_metadata
        WHERE table_name = 'customer_summary';
    """)

    _display = [
        mo.md("""
        **⏱️ Incremental Sync Pattern with Metadata Tracking:**

        **Sync Status:**
        """),
        mo.ui.table(sync_status_result),
        mo.md("""

        **🔄 Incremental Sync Benefits:**
        - **Efficiency**: Process only new/changed data
        - **Scalability**: Handle large datasets without full refresh
        - **Reliability**: Track sync status and recovery points
        - **Auditability**: Maintain processing history

        **🛠️ Implementation Pattern:**
        1. **Check Metadata**: Determine last sync timestamp
        2. **Filter Data**: Process only records newer than last sync
        3. **Apply Changes**: Update existing records incrementally
        4. **Update Metadata**: Record successful sync completion

        **💼 Real-World Applications:**
        - **ETL Pipelines**: Daily/hourly data updates
        - **Data Replication**: Sync between systems
        - **Change Data Capture**: Track and apply data changes
        """),
    ]
    return


@app.cell
def _(mo):
    """
    # Performance Optimization and Debugging

    Techniques for analyzing and optimizing SQL query performance.
    """
    mo.md("""
    ## ⚡ Performance Optimization and Debugging

    Learning to analyze query performance and identify optimization opportunities.
    """)
    return


@app.cell
def _(mo):
    """
    ### Query Execution Analysis
    """
    # Complex analytical query for performance analysis
    complex_query = """
    -- Complex analytical query for performance testing
    WITH customer_behavior AS (
        SELECT 
            customer_id,
            region,
            COUNT(*) as transaction_count,
            SUM(price * quantity) as total_spent,
            AVG(price * quantity) as avg_transaction,
            STDDEV(price * quantity) as transaction_stddev,
            MIN(date) as first_purchase,
            MAX(date) as last_purchase,
            COUNT(DISTINCT product_id) as unique_products,

            -- Advanced metrics
            SUM(CASE WHEN promo_code IS NOT NULL THEN 1 ELSE 0 END) as promo_transactions,
            SUM(CASE WHEN promo_code IS NOT NULL THEN price * quantity ELSE 0 END) as promo_revenue
        FROM transactions
        WHERE date >= '2024-01-01'
        GROUP BY customer_id, region
        HAVING COUNT(*) >= 3  -- Customers with 3+ transactions
    ),
    customer_segments AS (
        SELECT 
            *,
            CASE 
                WHEN total_spent >= 1000 THEN 'High Value'
                WHEN total_spent >= 500 THEN 'Medium Value' 
                ELSE 'Low Value'
            END as value_segment,

            CASE
                WHEN transaction_count >= 15 THEN 'High Frequency'
                WHEN transaction_count >= 8 THEN 'Medium Frequency'
                ELSE 'Low Frequency' 
            END as frequency_segment,

            -- Customer lifetime in days
            DATEDIFF('day', first_purchase, last_purchase) as customer_lifetime_days,

            -- Promotion usage rate
            ROUND(promo_transactions * 100.0 / transaction_count, 1) as promo_usage_rate
        FROM customer_behavior
    ),
    segment_analysis AS (
        SELECT 
            value_segment,
            frequency_segment,
            region,
            COUNT(*) as customers_in_segment,
            ROUND(AVG(total_spent), 2) as avg_segment_spending,
            ROUND(AVG(transaction_count), 1) as avg_segment_frequency,
            ROUND(AVG(customer_lifetime_days), 1) as avg_segment_lifetime,
            ROUND(AVG(promo_usage_rate), 1) as avg_promo_usage_rate,
            ROUND(SUM(total_spent), 2) as total_segment_revenue
        FROM customer_segments
        GROUP BY value_segment, frequency_segment, region
    )
    SELECT 
        value_segment,
        frequency_segment,
        region,
        customers_in_segment,
        avg_segment_spending,
        avg_segment_frequency,
        avg_segment_lifetime,
        avg_promo_usage_rate,
        total_segment_revenue,

        -- Segment performance metrics
        ROUND(total_segment_revenue / customers_in_segment, 2) as revenue_per_customer,
        RANK() OVER (ORDER BY total_segment_revenue DESC) as segment_revenue_rank,

        -- Percentage of total customers and revenue
        ROUND(customers_in_segment * 100.0 / SUM(customers_in_segment) OVER (), 2) as pct_of_customers,
        ROUND(total_segment_revenue * 100.0 / SUM(total_segment_revenue) OVER (), 2) as pct_of_revenue

    FROM segment_analysis
    ORDER BY total_segment_revenue DESC
    """

    # Execute the complex query
    performance_result = mo.sql(f"{complex_query}")

    _display = [
        mo.md("""
        **📊 Complex Customer Segmentation Analysis Results:**
        """),
        mo.ui.table(performance_result.head(15)),
        mo.md(f"""

        **Query Complexity Analysis:**
        - **CTEs Used**: 3 (customer_behavior, customer_segments, segment_analysis)
        - **Aggregation Levels**: Customer → Segment → Regional Summary
        - **Window Functions**: RANK(), SUM() OVER()
        - **Advanced Metrics**: Standard deviation, lifetime calculations, percentages
        - **Total Segments**: {len(performance_result)}

        **🎯 Performance Considerations:**
        - **Multi-level Aggregation**: Can be memory intensive
        - **Window Functions**: May require sorting operations  
        - **Complex JOINs**: Between multiple CTEs
        - **Statistical Functions**: STDDEV requires full data pass
        """),
    ]
    return (complex_query,)


@app.cell
def _(mo):
    """
    ### Query Optimization Strategies
    """
    optimization_examples = """
    **🚀 Query Optimization Strategies Demonstrated:**

    **1. Filtering Early:**
    ```sql
    -- ❌ Poor: Filter after aggregation
    SELECT customer_id, SUM(amount) 
    FROM large_table 
    GROUP BY customer_id
    HAVING SUM(amount) > 1000;

    -- ✅ Better: Filter before aggregation when possible
    SELECT customer_id, SUM(amount)
    FROM large_table 
    WHERE date >= '2024-01-01'  -- Reduce rows early
    GROUP BY customer_id
    HAVING SUM(amount) > 1000;
    ```

    **2. Index-Friendly Conditions:**
    ```sql
    -- ❌ Poor: Function on column prevents index usage
    WHERE EXTRACT(YEAR FROM date) = 2024

    -- ✅ Better: Range condition can use index
    WHERE date >= '2024-01-01' AND date < '2025-01-01'
    ```

    **3. Appropriate Aggregation Order:**
    ```sql
    -- ✅ Good: Group by most selective columns first
    GROUP BY customer_id, region, product_category
    -- customer_id is most selective (highest cardinality)
    ```

    **4. CTE vs Subquery Choice:**
    ```sql
    -- CTE: Good for readability, multiple references
    WITH customer_totals AS (SELECT ...)
    SELECT * FROM customer_totals WHERE ...
    UNION
    SELECT * FROM customer_totals WHERE ...

    -- Subquery: Often better for simple one-time use
    SELECT * FROM table 
    WHERE customer_id IN (SELECT customer_id FROM ...)
    ```
    """

    mo.md(f"""
    {optimization_examples}

    **⚡ Performance Best Practices Applied:**

    **In Our Complex Query:**
    1. **Early Filtering**: `WHERE date >= '2024-01-01'` reduces dataset size
    2. **HAVING Clause**: `HAVING COUNT(*) >= 3` filters after aggregation appropriately
    3. **Logical CTE Flow**: Each CTE builds on the previous one systematically
    4. **Window Function Efficiency**: Used only in final SELECT to minimize computation

    **📈 Monitoring Query Performance:**
    - **DuckDB**: Provides automatic query optimization
    - **PostgreSQL**: Use `EXPLAIN ANALYZE` for detailed execution plans  
    - **Key Metrics**: Execution time, memory usage, rows processed
    - **Optimization Focus**: Reduce data movement, leverage indexes, minimize sorts
    """)
    return


@app.cell
def _(complex_query, mo):
    """
    ### Debugging Complex Queries - Step-by-Step Approach
    """
    debugging_demo = """
    **🔍 Debugging Methodology Demonstrated:**

    **Step 1: Start Simple**
    ```sql
    -- Validate base data
    SELECT COUNT(*) FROM transactions WHERE date >= '2024-01-01';
    ```

    **Step 2: Build Incrementally** 
    ```sql
    -- Add basic aggregation
    SELECT customer_id, COUNT(*), SUM(price * quantity)
    FROM transactions 
    WHERE date >= '2024-01-01'
    GROUP BY customer_id
    LIMIT 10;
    ```

    **Step 3: Add Complexity Gradually**
    ```sql
    -- Add business logic
    WITH customer_base AS (
        SELECT customer_id, COUNT(*) as tx_count, SUM(price * quantity) as total
        FROM transactions WHERE date >= '2024-01-01'
        GROUP BY customer_id
    )
    SELECT *, 
           CASE WHEN total >= 1000 THEN 'High' ELSE 'Low' END as segment
    FROM customer_base;
    ```

    **Step 4: Validate Each Step**
    ```sql
    -- Check intermediate results make sense
    SELECT segment, COUNT(*), AVG(total), MIN(total), MAX(total)
    FROM customer_segments
    GROUP BY segment;
    ```
    """

    # Demonstrate validation query
    validation_query = (
        """
    -- Validation: Check our segment analysis makes sense
    SELECT 
        value_segment,
        frequency_segment,
        COUNT(*) as segment_count,
        ROUND(MIN(avg_segment_spending), 2) as min_spending,
        ROUND(MAX(avg_segment_spending), 2) as max_spending,
        ROUND(MIN(avg_segment_frequency), 1) as min_frequency,
        ROUND(MAX(avg_segment_frequency), 1) as max_frequency
    FROM ("""
        + complex_query
        + """) segment_results
    GROUP BY value_segment, frequency_segment
    ORDER BY 
        CASE value_segment 
            WHEN 'High Value' THEN 1 
            WHEN 'Medium Value' THEN 2 
            ELSE 3 
        END,
        CASE frequency_segment 
            WHEN 'High Frequency' THEN 1 
            WHEN 'Medium Frequency' THEN 2 
            ELSE 3 
        END;
    """
    )

    validation_result = mo.sql(f"{validation_query}")

    _display = [
        mo.md(f"""
        {debugging_demo}

        **✅ Validation Results from Our Complex Query:**
        """),
        mo.ui.table(validation_result),
        mo.md("""

        **🎯 Validation Insights:**
        - **Logical Segments**: High Value customers do have higher spending ranges
        - **Frequency Alignment**: High Frequency customers have higher transaction counts
        - **Data Quality**: No obvious outliers or data quality issues
        - **Business Logic**: Segmentation thresholds working as expected

        **🛠️ Debugging Toolkit:**
        - **COUNT(*)**: Always validate row counts at each step
        - **MIN/MAX/AVG**: Check ranges make business sense
        - **LIMIT**: Use small samples during development
        - **Intermediate CTEs**: Inspect each transformation step
        - **Data Profiling**: Understand your data distribution patterns
        """),
    ]
    return


@app.cell
def _(mo):
    """
    # Summary

    Wrapping up our comprehensive SQL learning journey.
    """
    mo.md("""
    ## 🎯 Learning Summary

    Congratulations! You've completed a comprehensive journey through advanced SQL concepts.
    """)
    return


@app.cell
def _(mo, taxi_summary_result, transactions):
    """
    ### What We've Covered
    """
    learning_summary = f"""
    **📚 Comprehensive Topics Mastered:**

    **1. Window Functions** 
    - ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE()
    - Running totals and moving averages
    - LAG/LEAD for time series analysis
    - Frame specifications and performance considerations

    **2. Advanced PostgreSQL Commands**
    - QUALIFY clause for filtering window functions
    - FILTER clause for conditional aggregations  
    - WITHIN GROUP for statistical calculations
    - Array and JSON operations

    **3. CTE vs Subquery Framework**
    - Performance trade-offs and decision matrix
    - Readability vs efficiency considerations
    - Complex multi-step analysis patterns

    **4. PIVOT/UNPIVOT Operations**
    - Manual pivot techniques with CASE statements
    - UNION ALL approach for unpivoting
    - Business applications for data transformation

    **5. Idempotent Operations**
    - UPSERT patterns with ON CONFLICT
    - Conditional update logic
    - Incremental sync patterns with metadata tracking
    - Data pipeline reliability techniques

    **6. Performance Optimization**
    - Query execution analysis
    - Index usage strategies  
    - Debugging complex queries step-by-step
    - Performance monitoring and validation

    **📊 Real Datasets Used:**
    - **Transaction Data**: {len(transactions):,} records with customer, product, and regional information
    - **NYC Taxi Data**: {taxi_summary_result.head(1).select("total_trips").item():,} trips with temporal and geographic patterns
    """

    mo.md(learning_summary)
    return


if __name__ == "__main__":
    app.run()
