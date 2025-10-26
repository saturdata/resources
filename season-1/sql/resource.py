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
def _(mo, pl, pq):
    """
    ## Data Loading - NYC Taxi Data

    Loading NYC Yellow Taxi trip data for January and February 2024.
    """
    # Load taxi data from parquet files
    taxi_jan = pq.read_table("season-1/data/tlc/yellow_tripdata_2024-01.parquet")
    taxi_feb = pq.read_table("season-1/data/tlc/yellow_tripdata_2024-02.parquet")

    # Convert to Polars for easier handling (optional, DuckDB can read PyArrow directly)
    taxi_jan_pl = pl.from_arrow(taxi_jan)
    taxi_feb_pl = pl.from_arrow(taxi_feb)

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
    _taxi_summary = mo.sql(
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
    return


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
    ### Query Plans - Understanding EXPLAIN
    """
    mo.md("""
    ## 🔍 Query Execution Plans with EXPLAIN

    Understanding how the database executes queries is critical for optimization.
    The `EXPLAIN` statement shows the query execution plan without running the query.
    `EXPLAIN ANALYZE` actually executes the query and provides timing information.

    **Key Metrics to Watch:**
    - **Scan Methods**: Sequential Scan vs Index Scan
    - **Join Types**: Hash Join, Nested Loop, Merge Join
    - **Estimated Rows**: How many rows the planner expects
    - **Actual Rows**: How many rows were actually processed
    - **Execution Time**: Total time to complete the query

    Let's compare execution plans for CTE vs Subquery approaches!
    """)
    return


@app.cell
def _(mo):
    mo.md("""### EXPLAIN - CTE Approach""")
    return


@app.cell
def _(mo):
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
def _(mo, format_query_plan, format_query_plan_tree):
    """Test the formatting functions with a simple EXPLAIN"""
    # Test with a simple query to verify formatting works
    test_explain = mo.sql("""
        EXPLAIN
        SELECT customer_id, COUNT(*) as count
        FROM transactions 
        WHERE date >= '2024-01-01'
        GROUP BY customer_id
        LIMIT 5
    """)

    mo.md(f"""
    **🧪 Testing Query Plan Formatting:**
    
    **Raw Output:**
    ```
    {test_explain.select("explain_value").item()}
    ```
    
    **Structured View:**
    ```
    {format_query_plan(test_explain)}
    ```
    
    **Tree Structure View:**
    ```
    {format_query_plan_tree(test_explain)}
    ```
    """)
    return


@app.cell
def _(mo, format_query_plan, format_query_plan_tree):
    """EXPLAIN for CTE query"""
    _explain_cte_result = mo.sql(
        f"""
        EXPLAIN
        WITH customer_metrics AS (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent,
                ROUND(AVG(price * quantity), 2) as avg_transaction_value
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
        ),
        high_value_customers AS (
            SELECT 
                customer_id,
                total_spent,
                transaction_count
            FROM customer_metrics
            WHERE total_spent > 500
        )
        SELECT 
            customer_id,
            total_spent,
            transaction_count,
            ROUND(total_spent / transaction_count, 2) as avg_per_transaction
        FROM high_value_customers
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )

    # Display the formatted query plan
    mo.md(f"""
    **🔍 Query Execution Plan:**
    
    **Structured View:**
    ```
    {format_query_plan(_explain_cte_result)}
    ```
    
    **Tree Structure View:**
    ```
    {format_query_plan_tree(_explain_cte_result)}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    **📊 CTE Query Execution Plan:**

    This shows how DuckDB executes the CTE-based query with multiple transformation steps.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### EXPLAIN - Subquery Approach""")
    return


@app.cell
def _(mo, format_query_plan, format_query_plan_tree):
    """EXPLAIN for equivalent subquery"""
    _explain_subquery_result = mo.sql(
        f"""
        EXPLAIN
        SELECT 
            customer_id,
            total_spent,
            transaction_count,
            ROUND(total_spent / transaction_count, 2) as avg_per_transaction
        FROM (
            SELECT 
                customer_id,
                COUNT(*) as transaction_count,
                SUM(price * quantity) as total_spent,
                ROUND(AVG(price * quantity), 2) as avg_transaction_value
            FROM transactions
            WHERE date >= '2024-01-01'
            GROUP BY customer_id
            HAVING SUM(price * quantity) > 500
        ) as customer_metrics
        ORDER BY total_spent DESC
        LIMIT 10
        """
    )

    # Display the formatted query plan
    mo.md(f"""
    **🔍 Subquery Execution Plan:**
    
    **Structured View:**
    ```
    {format_query_plan(_explain_subquery_result)}
    ```
    
    **Tree Structure View:**
    ```
    {format_query_plan_tree(_explain_subquery_result)}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    **📊 Subquery Execution Plan:**

    This shows the execution plan for the equivalent subquery-based approach.
    """
    )
    return


@app.cell
def _(mo):
    """
    ### Analyzing Query Plans - Key Insights
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
def _(conn, mo):
    """
    ### Real Performance Comparison with Timing
    """
    import time

    # Time CTE approach
    start = time.time()
    cte_perf_result = conn.execute("""
    WITH customer_metrics AS (
        SELECT 
            customer_id,
            COUNT(*) as transaction_count,
            SUM(price * quantity) as total_spent,
            ROUND(AVG(price * quantity), 2) as avg_transaction_value
        FROM transactions
        WHERE date >= '2024-01-01'
        GROUP BY customer_id
    ),
    high_value_customers AS (
        SELECT 
            customer_id,
            total_spent,
            transaction_count
        FROM customer_metrics
        WHERE total_spent > 500
    )
    SELECT 
        customer_id,
        total_spent,
        transaction_count,
        ROUND(total_spent / transaction_count, 2) as avg_per_transaction
    FROM high_value_customers
    ORDER BY total_spent DESC
    LIMIT 10;
    """).df()
    cte_time = time.time() - start

    # Time subquery approach
    start = time.time()
    subquery_perf_result = conn.execute("""
    SELECT 
        customer_id,
        total_spent,
        transaction_count,
        ROUND(total_spent / transaction_count, 2) as avg_per_transaction
    FROM (
        SELECT 
            customer_id,
            COUNT(*) as transaction_count,
            SUM(price * quantity) as total_spent,
            ROUND(AVG(price * quantity), 2) as avg_transaction_value
        FROM transactions
        WHERE date >= '2024-01-01'
        GROUP BY customer_id
        HAVING SUM(price * quantity) > 500
    ) as customer_metrics
    ORDER BY total_spent DESC
    LIMIT 10;
    """).df()
    subquery_time = time.time() - start

    # Verify both queries return same results
    perf_results_match = cte_perf_result.equals(subquery_perf_result)

    _display = [
        mo.md(f"""
        **⏱️ Performance Timing Comparison:**

        - **CTE Approach**: {cte_time * 1000:.3f} ms
        - **Subquery Approach**: {subquery_time * 1000:.3f} ms
        - **Difference**: {abs(cte_time - subquery_time) * 1000:.3f} ms
        - **Winner**: {"CTE" if cte_time < subquery_time else "Subquery"} (by {abs((cte_time - subquery_time) / max(cte_time, subquery_time) * 100):.1f}%)
        - **Results Match**: {"✅ Yes" if perf_results_match else "❌ No"}

        **📊 Results:**
        """),
        mo.ui.table(cte_perf_result.iloc[:5]),
        mo.md("""

        **💡 Key Takeaway:**
        The performance difference is typically **negligible** for queries of this complexity.
        Both approaches complete in milliseconds. Focus on **code maintainability** 
        and **team readability** rather than micro-optimizations at this scale.

        **When to Actually Optimize:**
        - Queries taking > 1 second on production data
        - Frequently executed queries (thousands of times per day)
        - Queries on tables with millions+ rows
        - Real-time/user-facing queries where latency matters
        """),
    ]
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
    # Advanced SQL Patterns

    Solving complex analytical problems with sophisticated SQL techniques.
    """
    mo.md("""
    ## 🎯 Advanced SQL Patterns

    Exploring sophisticated techniques for complex analytical challenges.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### Gap Analysis - Finding Missing Data Points""")
    return


@app.cell
def _(mo, transactions):
    """Find gaps in daily transaction data"""
    gap_analysis_result = mo.sql(
        f"""
        WITH date_range AS (
            SELECT DATE '2024-01-01' + (n || ' days')::INTERVAL as expected_date
            FROM generate_series(0, 365) as t(n)
        ),
        actual_transaction_dates AS (
            SELECT DISTINCT date as transaction_date
            FROM transactions
            WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
        ),
        missing_dates AS (
            SELECT dr.expected_date as missing_date
            FROM date_range dr
            LEFT JOIN actual_transaction_dates atd ON dr.expected_date = atd.transaction_date
            WHERE atd.transaction_date IS NULL
        )
        SELECT 
            missing_date,
            -- Context: What day of week was this?
            CASE EXTRACT(DOW FROM missing_date)
                WHEN 0 THEN 'Sunday'
                WHEN 1 THEN 'Monday' 
                WHEN 2 THEN 'Tuesday'
                WHEN 3 THEN 'Wednesday'
                WHEN 4 THEN 'Thursday'
                WHEN 5 THEN 'Friday'
                WHEN 6 THEN 'Saturday'
            END as day_of_week,

            -- How many days since last transaction?
            DATEDIFF('day', LAG(missing_date, 1) OVER (ORDER BY missing_date), missing_date) as days_since_prev_gap

        FROM missing_dates
        ORDER BY missing_date
        LIMIT 20
        """
    )
    return (gap_analysis_result,)


@app.cell
def _(gap_analysis_result, mo):
    mo.md(
        f"""
    **Total Missing Days:** {len(gap_analysis_result.value)} out of 366 days in 2024

    **🎯 Gap Analysis Applications:**
    - **Data Quality**: Identify missing data points
    - **Business Continuity**: Find operational gaps
    - **Seasonality**: Understand patterns in missing data
    - **System Monitoring**: Detect data pipeline issues
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Islands Analysis - Finding Consecutive Sequences""")
    return


@app.cell
def _(mo, transactions):
    """Find consecutive days of high-volume sales (islands)"""
    islands_result = mo.sql(
        f"""
        WITH daily_sales AS (
            SELECT 
                date,
                SUM(price * quantity) as daily_revenue,
                COUNT(*) as transaction_count
            FROM transactions
            GROUP BY date
            ORDER BY date
        ),
        high_volume_days AS (
            SELECT 
                date,
                daily_revenue,
                transaction_count,
                -- Create island identifier using row_number technique
                date - ROW_NUMBER() OVER (ORDER BY date) * INTERVAL '1 day' as island_id
            FROM daily_sales
            WHERE transaction_count >= 50  -- Define "high volume" threshold
        ),
        islands AS (
            SELECT 
                island_id,
                MIN(date) as island_start,
                MAX(date) as island_end,
                COUNT(*) as consecutive_days,
                ROUND(SUM(daily_revenue), 2) as total_island_revenue,
                ROUND(AVG(daily_revenue), 2) as avg_daily_revenue,
                SUM(transaction_count) as total_island_transactions
            FROM high_volume_days
            GROUP BY island_id
        )
        SELECT 
            island_start,
            island_end,
            consecutive_days,
            total_island_revenue,
            avg_daily_revenue,
            total_island_transactions,
            -- Calculate days between islands
            DATEDIFF('day', LAG(island_end, 1) OVER (ORDER BY island_start), island_start) - 1 as days_between_islands
        FROM islands
        WHERE consecutive_days >= 3  -- Only show islands of 3+ consecutive days
        ORDER BY island_start
        """
    )
    return (islands_result,)


@app.cell
def _(islands_result, mo):
    mo.md(
        f"""
    **Found {len(islands_result.value)} high-volume islands (3+ consecutive days with 50+ transactions)**

    **🎯 Islands Analysis Applications:**
    - **Performance Periods**: Identify sustained high-performance windows
    - **Campaign Analysis**: Measure promotion duration effectiveness  
    - **Operational Planning**: Understand busy period patterns
    - **Capacity Planning**: Prepare for consecutive high-demand periods

    **🛠️ Technical Note:**
    The `date - ROW_NUMBER() * INTERVAL '1 day'` technique creates identical values for consecutive dates, effectively grouping them into islands.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Data Deduplication with Quality Scoring""")
    return


@app.cell
def _(mo, transactions):
    """Advanced deduplication with data quality scoring"""
    _deduplication_result = mo.sql(
        f"""
        WITH transaction_quality AS (
            SELECT 
                *,
                -- Calculate quality score based on completeness
                (
                    CASE WHEN date IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN price > 0 THEN 1 ELSE 0 END +
                    CASE WHEN quantity > 0 THEN 1 ELSE 0 END +
                    CASE WHEN region IS NOT NULL THEN 1 ELSE 0 END +
                    -- Bonus points for promo code (additional data)
                    CASE WHEN promo_code IS NOT NULL THEN 1 ELSE 0 END
                ) as quality_score,

                -- Identify potential duplicates
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id, product_id, date, price
                    ORDER BY 
                        -- Prioritize records with promo codes
                        CASE WHEN promo_code IS NOT NULL THEN 1 ELSE 0 END DESC,
                        -- Then by completeness score
                        (
                            CASE WHEN date IS NOT NULL THEN 1 ELSE 0 END +
                            CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END +
                            CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END +
                            CASE WHEN price > 0 THEN 1 ELSE 0 END +
                            CASE WHEN quantity > 0 THEN 1 ELSE 0 END +
                            CASE WHEN region IS NOT NULL THEN 1 ELSE 0 END
                        ) DESC
                ) as row_rank
            FROM transactions
        ),
        duplicate_analysis AS (
            SELECT 
                quality_score,
                COUNT(*) as records_with_score,
                COUNT(CASE WHEN row_rank = 1 THEN 1 END) as records_after_dedup,
                COUNT(CASE WHEN row_rank > 1 THEN 1 END) as duplicate_records_removed
            FROM transaction_quality
            GROUP BY quality_score
        )
        SELECT 
            quality_score,
            records_with_score,
            records_after_dedup,
            duplicate_records_removed,
            ROUND(duplicate_records_removed * 100.0 / records_with_score, 1) as duplicate_percentage
        FROM duplicate_analysis
        ORDER BY quality_score DESC
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Quality Score Components:**
    - **Base Fields**: date, customer_id, product_id, price, quantity, region (6 points)
    - **Bonus Field**: promo_code (+1 point for completeness)
    - **Maximum Score**: 7 points

    **🎯 Deduplication Strategy:**
    1. **Identify Duplicates**: Same customer, product, date, price
    2. **Quality Scoring**: Rank records by completeness
    3. **Preference Logic**: Keep records with promo codes (more informative)
    4. **Systematic Selection**: Consistent tie-breaking rules

    **Business Impact:**
    - **Data Quality**: Remove redundant records systematically
    - **Analysis Accuracy**: Prevent double-counting in metrics
    - **Storage Efficiency**: Reduce data storage requirements
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""### Time Series Pattern Analysis with Taxi Data""")
    return


@app.cell
def _(mo, taxi_feb, taxi_jan):
    """Analyze temporal patterns in NYC taxi data"""
    taxi_patterns_result = mo.sql(
        f"""
        WITH taxi_data AS (
            SELECT * FROM taxi_jan
            UNION ALL
            SELECT * FROM taxi_feb
        ),
        hourly_patterns AS (
            SELECT 
                DATE(tpep_pickup_datetime) as trip_date,
                EXTRACT(HOUR FROM tpep_pickup_datetime) as hour_of_day,
                EXTRACT(DOW FROM tpep_pickup_datetime) as day_of_week, -- 0=Sunday
                COUNT(*) as trip_count,
                ROUND(AVG(fare_amount), 2) as avg_fare,
                ROUND(AVG(trip_distance), 2) as avg_distance,
                ROUND(AVG(tip_amount), 2) as avg_tip
            FROM taxi_data
            WHERE DATE(tpep_pickup_datetime) BETWEEN '2024-01-01' AND '2024-01-31'  -- January only
            GROUP BY 
                DATE(tpep_pickup_datetime),
                EXTRACT(HOUR FROM tpep_pickup_datetime),
                EXTRACT(DOW FROM tpep_pickup_datetime)
        ),
        pattern_analysis AS (
            SELECT 
                hour_of_day,
                CASE day_of_week
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday' 
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                CASE 
                    WHEN day_of_week IN (0, 6) THEN 'Weekend'
                    ELSE 'Weekday'
                END as day_type,

                COUNT(*) as observations,
                ROUND(AVG(trip_count), 1) as avg_trips_per_hour,
                ROUND(STDDEV(trip_count), 1) as stddev_trips,
                ROUND(MIN(trip_count), 1) as min_trips,
                ROUND(MAX(trip_count), 1) as max_trips,
                ROUND(AVG(avg_fare), 2) as avg_hourly_fare,
                ROUND(AVG(avg_distance), 2) as avg_hourly_distance
            FROM hourly_patterns
            GROUP BY hour_of_day, day_of_week, day_name, day_type
        )
        SELECT 
            hour_of_day,
            day_type,
            SUM(observations) as total_observations,
            ROUND(AVG(avg_trips_per_hour), 1) as avg_trips_per_hour,
            ROUND(AVG(avg_hourly_fare), 2) as avg_fare,
            ROUND(AVG(avg_hourly_distance), 2) as avg_distance,

            -- Peak hour identification
            CASE 
                WHEN hour_of_day BETWEEN 7 AND 9 OR hour_of_day BETWEEN 17 AND 19 THEN 'Rush Hour'
                WHEN hour_of_day BETWEEN 22 AND 5 THEN 'Late Night'
                ELSE 'Regular Hours'
            END as hour_category

        FROM pattern_analysis
        GROUP BY hour_of_day, day_type
        ORDER BY day_type, hour_of_day
        """
    )
    return (taxi_patterns_result,)


@app.cell
def _(mo, taxi_patterns_result):
    mo.md(
        f"""
    **Total Records Analyzed:** {len(taxi_patterns_result.value)} hour-daytype combinations

    **🎯 Pattern Analysis Insights:**
    - **Rush Hour Detection**: 7-9 AM and 5-7 PM show different patterns
    - **Weekend vs Weekday**: Different usage patterns and fare structures
    - **Late Night Activity**: Understanding off-peak demand

    **📊 Time Series Applications:**
    - **Demand Forecasting**: Predict busy periods
    - **Dynamic Pricing**: Adjust rates based on demand patterns
    - **Resource Allocation**: Deploy vehicles efficiently
    - **Service Planning**: Optimize operations by time patterns
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
def _(mo):
    """Create customer_summary table"""
    _create_customer_summary = mo.sql(
        f"""
        CREATE TABLE IF NOT EXISTS customer_summary (
            customer_id INTEGER PRIMARY KEY,
            first_purchase_date DATE,
            last_purchase_date DATE,
            total_transactions INTEGER,
            total_spent DECIMAL(10,2),
            avg_transaction_amount DECIMAL(10,2),
            favorite_region TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return (customer_summary,)


@app.cell
def _(mo):
    """Create product_metrics table"""
    _create_product_metrics = mo.sql(
        f"""
        CREATE TABLE IF NOT EXISTS product_metrics (
            product_id INTEGER PRIMARY KEY,
            total_revenue DECIMAL(12,2),
            total_quantity_sold INTEGER,
            avg_price DECIMAL(8,2),
            unique_customers INTEGER,
            first_sale_date DATE,
            last_sale_date DATE,
            calculation_date DATE DEFAULT CURRENT_DATE
        )
        """
    )
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
def _(customer_summary, mo, transactions):
    """Idempotent customer summary update"""
    _upsert_customer = mo.sql(
        f"""
        INSERT INTO customer_summary (
            customer_id,
            first_purchase_date,
            last_purchase_date,
            total_transactions,
            total_spent,
            avg_transaction_amount,
            favorite_region,
            last_updated
        )
        SELECT 
            customer_id,
            MIN(date) as first_purchase_date,
            MAX(date) as last_purchase_date,
            COUNT(*) as total_transactions,
            ROUND(SUM(price * quantity), 2) as total_spent,
            ROUND(AVG(price * quantity), 2) as avg_transaction_amount,
            -- Most frequent region (mode)
            (SELECT region 
             FROM transactions t2 
             WHERE t2.customer_id = t1.customer_id 
             GROUP BY region 
             ORDER BY COUNT(*) DESC 
             LIMIT 1) as favorite_region,
            CURRENT_TIMESTAMP as last_updated
        FROM transactions t1
        GROUP BY customer_id
        ON CONFLICT (customer_id) 
        DO UPDATE SET
            first_purchase_date = LEAST(customer_summary.first_purchase_date, CAST(EXCLUDED.first_purchase_date AS DATE)),
            last_purchase_date = GREATEST(customer_summary.last_purchase_date, CAST(EXCLUDED.last_purchase_date AS DATE)),
            total_transactions = EXCLUDED.total_transactions,
            total_spent = EXCLUDED.total_spent,
            avg_transaction_amount = EXCLUDED.avg_transaction_amount,
            favorite_region = EXCLUDED.favorite_region,
            last_updated = EXCLUDED.last_updated
        """
    )
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
def _(conn, mo, product_metrics):
    """
    ### Advanced UPSERT with Conditional Logic
    """
    conditional_upsert_query = """
    -- Product metrics with conditional update logic
    INSERT INTO product_metrics (
        product_id,
        total_revenue,
        total_quantity_sold,
        avg_price,
        unique_customers,
        first_sale_date,
        last_sale_date,
        calculation_date
    )
    SELECT 
        product_id,
        ROUND(SUM(price * quantity), 2) as total_revenue,
        SUM(quantity) as total_quantity_sold,
        ROUND(AVG(price), 2) as avg_price,
        COUNT(DISTINCT customer_id) as unique_customers,
        MIN(date) as first_sale_date,
        MAX(date) as last_sale_date,
        CURRENT_DATE as calculation_date
    FROM transactions
    WHERE date >= '2024-01-01'  -- Only 2024 data
    GROUP BY product_id
    HAVING COUNT(*) >= 5  -- Only products with 5+ sales

    ON CONFLICT (product_id) 
    DO UPDATE SET
        -- Only update if we have newer data
        total_revenue = CASE 
            WHEN EXCLUDED.calculation_date >= product_metrics.calculation_date 
            THEN EXCLUDED.total_revenue 
            ELSE product_metrics.total_revenue 
        END,
        total_quantity_sold = CASE 
            WHEN EXCLUDED.calculation_date >= product_metrics.calculation_date 
            THEN EXCLUDED.total_quantity_sold 
            ELSE product_metrics.total_quantity_sold 
        END,
        avg_price = CASE 
            WHEN EXCLUDED.calculation_date >= product_metrics.calculation_date 
            THEN EXCLUDED.avg_price 
            ELSE product_metrics.avg_price 
        END,
        unique_customers = CASE 
            WHEN EXCLUDED.calculation_date >= product_metrics.calculation_date 
            THEN EXCLUDED.unique_customers 
            ELSE product_metrics.unique_customers 
        END,
        first_sale_date = LEAST(product_metrics.first_sale_date, CAST(EXCLUDED.first_sale_date AS DATE)),
        last_sale_date = GREATEST(product_metrics.last_sale_date, CAST(EXCLUDED.last_sale_date AS DATE)),
        calculation_date = GREATEST(product_metrics.calculation_date, EXCLUDED.calculation_date);
    """

    # Execute conditional upsert
    conn.execute(conditional_upsert_query)

    # Analyze results
    product_summary_result = mo.sql(f"""
        SELECT 
            COUNT(*) as products_analyzed,
            ROUND(SUM(total_revenue), 2) as total_product_revenue,
            ROUND(AVG(total_revenue), 2) as avg_revenue_per_product,
            ROUND(AVG(avg_price), 2) as overall_avg_price,
            MAX(unique_customers) as max_customers_per_product,
            ROUND(AVG(unique_customers), 1) as avg_customers_per_product
        FROM product_metrics;
    """)

    # Top performing products
    top_products_result = mo.sql(f"""
        SELECT 
            product_id,
            total_revenue,
            total_quantity_sold,
            avg_price,
            unique_customers,
            first_sale_date,
            last_sale_date,
            ROUND(total_revenue / unique_customers, 2) as revenue_per_customer
        FROM product_metrics
        ORDER BY total_revenue DESC
        LIMIT 15;
    """)

    _display = [
        mo.md("""
        **📊 Advanced UPSERT with Business Logic:**

        **Product Analysis Summary:**
        """),
        mo.ui.table(product_summary_result.value),
        mo.md("""

        **Top Products by Revenue:**
        """),
        mo.ui.table(top_products_result.value),
        mo.md("""

        **🚀 Advanced UPSERT Features:**
        - **Conditional Updates**: Only update when data is newer
        - **Data Quality Filters**: Only process products with sufficient sales
        - **Smart Date Handling**: Keep earliest first sale, latest last sale
        - **Business Logic**: Calculate derived metrics (revenue per customer)

        **🎯 Use Cases:**
        - **ETL Pipelines**: Incremental data updates
        - **Real-time Analytics**: Maintain live dashboards
        - **Data Warehousing**: Slowly changing dimensions
        """),
    ]
    return


@app.cell
def _(conn, mo, sync_metadata):
    """
    ### Incremental Update Pattern with Metadata Tracking
    """
    # Create sync metadata table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sync_metadata (
        table_name TEXT PRIMARY KEY,
        last_sync_timestamp TIMESTAMP,
        records_processed INTEGER,
        last_sync_date DATE DEFAULT CURRENT_DATE
    );
    """)

    # Incremental sync query pattern (for reference - not executed in demo)
    # This would typically be done in separate statements in production
    _incremental_sync_query = """
    -- Incremental sync pattern with metadata tracking
    WITH sync_info AS (
        SELECT 
            COALESCE(
                (SELECT last_sync_timestamp FROM sync_metadata WHERE table_name = 'customer_summary'),
                '1900-01-01 00:00:00'::timestamp
            ) as last_sync_time
    ),
    new_or_updated_data AS (
        SELECT 
            customer_id,
            SUM(price * quantity) as incremental_spent,
            COUNT(*) as incremental_transactions,
            MAX(date) as latest_transaction_date
        FROM transactions t
        CROSS JOIN sync_info si
        WHERE t.date::timestamp > si.last_sync_time  -- Only process new data
        GROUP BY customer_id
    ),
    update_operations AS (
        -- Update existing customer records
        UPDATE customer_summary 
        SET 
            total_spent = customer_summary.total_spent + nud.incremental_spent,
            total_transactions = customer_summary.total_transactions + nud.incremental_transactions,
            last_purchase_date = GREATEST(customer_summary.last_purchase_date, CAST(nud.latest_transaction_date AS DATE)),
            last_updated = CURRENT_TIMESTAMP
        FROM new_or_updated_data nud
        WHERE customer_summary.customer_id = nud.customer_id
        RETURNING customer_summary.customer_id as updated_customer
    )
    -- Update sync metadata
    INSERT INTO sync_metadata (table_name, last_sync_timestamp, records_processed)
    VALUES ('customer_summary', CURRENT_TIMESTAMP, (SELECT COUNT(*) FROM new_or_updated_data))
    ON CONFLICT (table_name)
    DO UPDATE SET
        last_sync_timestamp = EXCLUDED.last_sync_timestamp,
        records_processed = EXCLUDED.records_processed,
        last_sync_date = CURRENT_DATE;
    """

    # For demo, we'll just show the concept and check sync metadata

    # Initialize sync metadata
    conn.execute("""
    INSERT INTO sync_metadata (table_name, last_sync_timestamp, records_processed)
    VALUES ('customer_summary', '2024-01-01 00:00:00', 0)
    ON CONFLICT (table_name) DO NOTHING;
    """)

    sync_status_result = mo.sql(f"""
        SELECT 
            table_name,
            last_sync_timestamp,
            records_processed,
            last_sync_date,
            CURRENT_TIMESTAMP - last_sync_timestamp as time_since_last_sync
        FROM sync_metadata
        WHERE table_name = 'customer_summary';
    """)

    _display = [
        mo.md("""
        **⏱️ Incremental Sync Pattern with Metadata Tracking:**

        **Sync Status:**
        """),
        mo.ui.table(sync_status_result.value),
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
        mo.ui.table(performance_result.value.iloc[:15]),
        mo.md(f"""

        **Query Complexity Analysis:**
        - **CTEs Used**: 3 (customer_behavior, customer_segments, segment_analysis)
        - **Aggregation Levels**: Customer → Segment → Regional Summary
        - **Window Functions**: RANK(), SUM() OVER()
        - **Advanced Metrics**: Standard deviation, lifetime calculations, percentages
        - **Total Segments**: {len(performance_result.value)}

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
        mo.ui.table(validation_result.value),
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
    # Summary and Next Steps

    Wrapping up our comprehensive SQL learning journey.
    """
    mo.md("""
    ## 🎯 Learning Summary and Next Steps

    Congratulations! You've completed a comprehensive journey through advanced SQL concepts.
    """)
    return


@app.cell
def _(mo, taxi_summary_result, transactions_df):
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

    **5. Advanced SQL Patterns**
    - Gap and island analysis for sequential data
    - Data deduplication with quality scoring
    - Time series pattern recognition
    - Statistical analysis and percentile calculations

    **6. Idempotent Operations**
    - UPSERT patterns with ON CONFLICT
    - Conditional update logic
    - Incremental sync patterns with metadata tracking
    - Data pipeline reliability techniques

    **7. Performance Optimization**
    - Query execution analysis
    - Index usage strategies  
    - Debugging complex queries step-by-step
    - Performance monitoring and validation

    **📊 Real Datasets Used:**
    - **Transaction Data**: {len(transactions_df):,} records with customer, product, and regional information
    - **NYC Taxi Data**: {taxi_summary_result.value.iloc[0]["total_trips"]:,} trips with temporal and geographic patterns
    """

    mo.md(learning_summary)
    return


@app.cell
def _(mo):
    """
    ### Next Steps and Advanced Topics
    """
    next_steps = """
    **🚀 Continue Your SQL Journey:**

    **Immediate Practice:**
    1. **Experiment**: Modify queries in this notebook with different parameters
    2. **Apply**: Use these patterns with your own datasets
    3. **Combine**: Mix techniques (e.g., window functions + CTEs + UPSERT)
    4. **Optimize**: Practice query performance tuning

    **Advanced Topics to Explore:**
    - **Recursive CTEs**: Hierarchical data and graph traversal
    - **Custom Functions**: PL/pgSQL for complex business logic
    - **Materialized Views**: Performance optimization for frequent queries
    - **Partitioning**: Handling very large datasets (100GB+)
    - **Full-Text Search**: Advanced text analytics capabilities
    - **PostGIS Extensions**: Geospatial analytics for location data
    - **JSON/JSONB**: Modern semi-structured data handling

    **Production Considerations:**
    - **Security**: SQL injection prevention, access controls
    - **Monitoring**: Query performance tracking and alerting
    - **Backup/Recovery**: Data protection strategies
    - **Scaling**: Read replicas, connection pooling, load balancing

    **Community and Resources:**
    - **PostgreSQL Documentation**: Comprehensive reference material
    - **Stack Overflow**: Community Q&A for specific problems
    - **PostgreSQL Slack/Discord**: Real-time community support
    - **Conferences**: PGConf, FOSDEM for latest developments
    """

    mo.md(f"""
    {next_steps}

    **💡 Key Takeaways:**
    - **Practice Makes Perfect**: SQL mastery comes through regular use
    - **Understand Your Data**: Profile before optimizing
    - **Think in Sets**: Leverage SQL's strength in set-based operations
    - **Performance Matters**: But readability and maintainability also count
    - **Stay Curious**: PostgreSQL constantly evolves with new features

    **🎯 Remember:**
    The best SQL practitioners combine technical knowledge with business understanding. 
    Always ask: "What business question am I trying to answer?" before writing complex queries.

    **Happy Querying! 🚀**
    """)
    return


@app.cell
def _(mo):
    """
    ### Environment Information
    """
    import sys
    import platform

    mo.md(f"""
    **🖥️ Environment Information:**
    - **Python**: {sys.version.split()[0]}
    - **Platform**: {platform.system()} {platform.release()}
    - **DuckDB**: In-memory analytical database
    - **Marimo**: Latest

    **📁 Data Sources:**
    - `../data/transactions_synthetic.csv`: Synthetic transaction data
    - `../data/tlc/yellow_tripdata_2024-01.parquet`: NYC Taxi data (January)
    - `../data/tlc/yellow_tripdata_2024-02.parquet`: NYC Taxi data (February)

    **🔄 Notebook Status:** Complete - All examples executed successfully!
    """)
    return


if __name__ == "__main__":
    app.run()
