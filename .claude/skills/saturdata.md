# Saturdata Project - Technical Skills Guide for AI Agents

## Project Overview

**Purpose**: Create interactive Marimo notebooks as companion resources to Saturdata podcast episodes

**Target Audience**: Early-career data professionals (0-3 years experience) transitioning into data engineering and analytics roles

**Learning Philosophy**: Discovery tools, not tutorials. Spark curiosity, build confidence, and create community connections. Users should feel like they learned WITH Shifra and Sam, not FROM them. See `style.md` for complete educational content guidelines.

**Episode Structure**: Progressive complexity across 6 focused topics (SQL, Polars/Pandas, Data Transformation, Visualization, Statistical Testing, Terminal/Git)

## Technical Stack

### Core Dependencies

```toml
[project]
name = "saturdata-resources"
requires-python = ">=3.11"
dependencies = [
    "marimo>=0.17.0",        # Interactive reactive notebooks
    "numpy>=1.24.0",          # Numerical computing
    "pandas>=2.0.0",          # Tabular data (baseline comparison)
    "polars[pyarrow]>=0.20",  # Modern data processing
    "scipy>=1.10.0",          # Statistical functions
    "matplotlib>=3.7.0",      # Low-level visualization
    "seaborn>=0.12.0",        # Statistical visualization
    "plotly>=6.3.1",          # Interactive web visualizations
    "pyarrow>=21.0.0",        # Columnar storage format
    "duckdb>=1.4.1",          # In-memory SQL analytics
    "sqlglot>=27.26.0",       # SQL parsing/generation
    "pyzmq>=27.1.0",          # ZeroMQ support (marimo backend)
]
```

### Environment Setup
- **Python**: 3.11+ (significant performance improvements)
- **Package Manager**: `uv` (faster, simpler than pip)
- **Installation**: `uv sync` from project root

### Library Usage Patterns

| Library | Primary Use Case | When to Use |
|---------|------------------|-------------|
| **Marimo** | Interactive notebooks | All episode notebooks |
| **DuckDB** | In-memory SQL analytics | SQL operations, large dataset queries |
| **Polars** | High-performance data processing | ETL pipelines, large datasets |
| **Pandas** | Baseline comparison, visualization prep | Converting Polars for Seaborn/Matplotlib |
| **Seaborn** | Statistical visualization | Business dashboards, publication-quality plots |
| **Plotly** | Interactive web visualizations | Exploratory analysis, presentations |

## Directory Structure

```
saturdata/resources/
├── .claude/
│   └── skills/
│       ├── style.md          # Educational content guidelines
│       └── saturdata.md      # This file - technical patterns
├── season-1/
│   ├── data/                 # Shared datasets for all episodes
│   │   ├── tlc/              # NYC Yellow Taxi data (Parquet)
│   │   │   ├── yellow_tripdata_2024-01.parquet
│   │   │   └── yellow_tripdata_2024-02.parquet
│   │   └── transactions_synthetic.csv  # 5M e-commerce transactions
│   ├── polars-pandas/        # Episode 1: Performance comparison
│   │   ├── resource.py       # Main Marimo notebook
│   │   ├── README.md         # Setup and context
│   │   ├── overview.md       # Reference material
│   │   └── data/             # Episode-specific generated data
│   ├── sql/                  # SQL for data professionals
│   ├── data-transformation/  # NumPy, Pandas, Polars patterns
│   ├── data-visualization/   # Seaborn, Plotly, Matplotlib
│   ├── statistical-testing/  # SciPy hypothesis testing
│   └── terminal/             # CLI and Git fundamentals
├── pyproject.toml            # Unified dependencies
├── uv.lock                   # Dependency lock file
└── README.md                 # Repository guide
```

### Directory Naming Conventions
- Lowercase with hyphens (e.g., `polars-pandas`, not `polars_pandas`)
- Descriptive episode/topic names
- `resource.py` for main notebook (consistent across all episodes)
- `data/` for shared datasets at season level

## Marimo Notebook Patterns

### Standard Cell Structure

Every Marimo notebook follows this pattern:

```python
import marimo

__generated_with = "0.18.0"  # Track marimo version
app = marimo.App(width="medium")  # Readable cell widths


@app.cell
def _(mo):
    """
    Opening markdown cell with episode context and learning objectives
    """
    mo.md(r"""
    # Episode Title

    **Episode Connection**: Brief reference to podcast discussion

    ## What You'll Discover
    - Learning objective 1
    - Learning objective 2
    - Learning objective 3

    **No Code Context Required**: This notebook can be fully understood without listening to the episode.
    """)
    return


@app.cell
def _():
    """
    Import cell - all package imports in one place
    """
    import marimo as mo
    import pandas as pd
    import polars as pl
    import duckdb
    from pathlib import Path
    return duckdb, mo, Path, pd, pl


@app.cell
def _(Path):
    """
    Path setup cell - critical for data loading reliability
    """
    # Handle both local and notebook contexts
    NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    DATA_DIR = NOTEBOOK_DIR.parent / "data"
    return DATA_DIR, NOTEBOOK_DIR


@app.cell
def _(DATA_DIR, pl):
    """
    Data loading cell - use absolute paths always
    """
    # Load with absolute path
    transactions = pl.read_csv(str(DATA_DIR / "transactions_synthetic.csv"))

    # Type conversions
    transactions = transactions.with_columns(
        pl.col("date").str.to_date().alias("date")
    )
    return (transactions,)  # Note the tuple return for single variable


@app.cell
def _(mo):
    """
    Educational markdown cells use narrative tone
    """
    mo.md("""
    ## Section Title

    Educational content in conversational tone. Reference hosts naturally:
    "Sam often encounters this pattern when..." or "Shifra's favorite approach..."

    **Key Concept**: Explain technical terms in plain English.
    """)
    return


@app.cell
def _(transactions):
    """
    Data processing cell with clear variable names
    """
    # Progressive complexity - start simple
    result = transactions.group_by("region").agg(
        pl.col("price").sum().alias("total_revenue")
    )
    return (result,)


# Always end with app.run() for Marimo execution
if __name__ == "__main__":
    app.run()
```

### File Path Handling (Complete Pattern)

```python
from pathlib import Path

# Standard pattern used in all notebooks
NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
DATA_DIR = NOTEBOOK_DIR.parent / "data"

# Load shared dataset
transactions_path = DATA_DIR / "transactions_synthetic.csv"
transactions = pl.read_csv(str(transactions_path))

# Load episode-specific data
episode_data_dir = NOTEBOOK_DIR / "data"
customers_path = episode_data_dir / "customers.csv"

# NYC Taxi data (Parquet)
taxi_jan_path = DATA_DIR / "tlc" / "yellow_tripdata_2024-01.parquet"
taxi_jan = pq.read_table(str(taxi_jan_path))
```

**Critical Rules**:
- Always use absolute paths via `Path`
- Convert `Path` to string when passing to libraries (`str(path)`)
- `NOTEBOOK_DIR` = notebook location, `DATA_DIR` = shared season data
- Check if data exists before loading for better error messages

### DuckDB Integration Pattern

```python
import duckdb
import marimo as mo

# Create in-memory connection
conn = duckdb.connect(":memory:")

# Optional: Set memory limits
conn.execute("SET memory_limit='2GB'")

# Load Polars DataFrame into DuckDB (automatic)
# DuckDB can query Polars DataFrames directly
transactions_df = pl.read_csv("path/to/data.csv")

# Execute SQL with mo.sql() - automatically discovers conn
result = mo.sql(f"""
    SELECT
        region,
        COUNT(*) as transaction_count,
        SUM(price * quantity) as total_revenue
    FROM transactions_df
    WHERE date >= '2024-01-01'
    GROUP BY region
    ORDER BY total_revenue DESC
""")

# result.value contains the DataFrame
# Result automatically displayed as interactive table in Marimo
```

**Key Features**:
- `mo.sql()` discovers `conn` variable automatically
- Supports f-string interpolation for Python variables
- Can query Polars DataFrames, Pandas DataFrames, Arrow tables directly
- Results have `.value` attribute containing DataFrame

### Interactive Element Patterns

```python
# Interactive table with filtering
result_table = mo.ui.table(dataframe)

# Altair chart integration
import altair as alt
chart = alt.Chart(dataframe).mark_point().encode(
    x='column1',
    y='column2'
)
mo.ui.altair_chart(chart)

# Markdown with inline code
mo.md(f"""
**Data Summary**:
- Total Records: {len(dataframe):,}
- Date Range: {dataframe["date"].min()} to {dataframe["date"].max()}
""")
```

## Data Handling Standards

### Dataset Requirements
- **Size**: Max 1000 rows for interactive performance (use sampling for larger datasets)
- **Realism**: Include messy data (nulls, duplicates, inconsistencies) for learning
- **Relatability**: Use business scenarios relevant to early-career professionals
- **Generation**: Include data generation code when possible (avoid external downloads)
- **Paths**: Always use absolute paths for reliability

### Shared Data Assets

**Transactions Synthetic Dataset** (`data/transactions_synthetic.csv`):
```
Columns: date, customer_id, product_id, quantity, price, region, promo_code
Rows: 5 million records (~330MB)
Purpose: Performance benchmarking, SQL learning, aggregation exercises
Date Range: 2024-01-01 to 2024-12-31
Regions: North, South, East, West
Promo Codes: PROMO10, PROMO20, PROMO30, NULL
```

**NYC Yellow Taxi Data** (`data/tlc/yellow_tripdata_2024-{01,02}.parquet`):
```
Format: Parquet (columnar, compressed)
Columns: tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
         trip_distance, fare_amount, total_amount, payment_type, etc.
Purpose: Real-world SQL learning, window functions, temporal analysis
```

### Data Loading Pattern (Complete Example)

```python
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq

# Path setup
NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
DATA_DIR = NOTEBOOK_DIR.parent / "data"

# CSV loading with type handling
transactions_path = DATA_DIR / "transactions_synthetic.csv"
transactions = pl.read_csv(str(transactions_path))

# Type conversions
transactions = transactions.with_columns([
    pl.col("date").str.to_date().alias("date"),
    pl.col("price").cast(pl.Float64),
    pl.col("quantity").cast(pl.Int32)
])

# Parquet loading (PyArrow)
taxi_path = DATA_DIR / "tlc" / "yellow_tripdata_2024-01.parquet"
taxi_jan = pq.read_table(str(taxi_path))

# Validation (optional but recommended)
assert transactions.height > 0, f"No data loaded from {transactions_path}"
assert "date" in transactions.columns, "Missing required 'date' column"

# Display summary in marimo
mo.md(f"""
**Dataset Loaded Successfully**:
- Records: {transactions.height:,}
- Date Range: {transactions["date"].min()} to {transactions["date"].max()}
- Regions: {", ".join(transactions["region"].unique().to_list())}
""")
```

## Common Code Patterns

### Benchmarking Pattern (Complete Function)

```python
import time
import tracemalloc
import psutil
from typing import Callable, Any

def benchmark_operation(func: Callable[[], Any], label: str) -> dict:
    """
    Benchmark function execution with time and memory tracking.

    Args:
        func: Callable that executes the operation to benchmark
        label: Descriptive label for the operation (e.g., "Pandas GroupBy")

    Returns:
        dict with keys: library, time_seconds, memory_mb, peak_memory_mb, result
    """
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    mem_before = process.memory_info().rss

    # Execute and time the function
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    # Memory measurements
    mem_after = process.memory_info().rss
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "library": label,
        "time_seconds": round(elapsed, 2),
        "memory_mb": round((mem_after - mem_before) / 1024**2, 1),
        "peak_memory_mb": round(peak / 1024**2, 1),
        "result": result
    }

# Usage example
results = []

# Benchmark Pandas
pandas_result = benchmark_operation(
    lambda: df_pandas.groupby("region")["price"].sum(),
    "Pandas"
)
results.append(pandas_result)

# Benchmark Polars
polars_result = benchmark_operation(
    lambda: df_polars.group_by("region").agg(pl.col("price").sum()),
    "Polars"
)
results.append(polars_result)

# Display comparison
comparison_df = pl.DataFrame(results)
mo.ui.table(comparison_df)
```

### Library Conversion Pattern

```python
# Process with Polars (fast for large data)
polars_result = large_df.group_by("region", "product_id").agg([
    pl.col("price").mean().alias("avg_price"),
    pl.col("quantity").sum().alias("total_quantity")
])

# Convert to Pandas only for visualization
pandas_df = polars_result.to_pandas()

# Visualize with Seaborn (requires Pandas)
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=pandas_df,
    x="avg_price",
    y="total_quantity",
    hue="region"
)
plt.title("Price vs Quantity by Region")
plt.show()

# Or use Plotly (works with Pandas)
import plotly.express as px
fig = px.scatter(
    pandas_df,
    x="avg_price",
    y="total_quantity",
    color="region",
    title="Interactive Price vs Quantity"
)
fig.show()
```

### SQL Query Pattern (DuckDB with Marimo)

```python
import duckdb
import marimo as mo

# Connection setup
conn = duckdb.connect(":memory:")

# Load data (DuckDB can query Polars/Pandas/Arrow directly)
transactions = pl.read_csv("path/to/data.csv")

# Simple query
result = mo.sql(f"""
    SELECT
        region,
        COUNT(*) as transaction_count,
        ROUND(AVG(price * quantity), 2) as avg_transaction_amount
    FROM transactions
    WHERE date >= '2024-01-01'
    GROUP BY region
    ORDER BY avg_transaction_amount DESC
""")

# Complex query with window functions
monthly_rankings = mo.sql(f"""
    WITH monthly_sales AS (
        SELECT
            DATE_TRUNC('month', date) as month,
            region,
            SUM(price * quantity) as monthly_revenue
        FROM transactions
        GROUP BY DATE_TRUNC('month', date), region
    )
    SELECT
        month,
        region,
        monthly_revenue,
        RANK() OVER (PARTITION BY month ORDER BY monthly_revenue DESC) as region_rank,
        SUM(monthly_revenue) OVER (PARTITION BY region ORDER BY month) as cumulative_revenue
    FROM monthly_sales
    ORDER BY month, region_rank
""")

# Results automatically displayed as interactive tables
```

### Visualization Patterns

```python
# Seaborn - Statistical visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Set publication-quality style
sns.set_theme(style="whitegrid")

# Multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot
sns.scatterplot(data=df, x="price", y="quantity", hue="region", ax=axes[0, 0])
axes[0, 0].set_title("Price vs Quantity by Region")

# Distribution plot
sns.histplot(data=df, x="price", kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Price Distribution")

# Box plot
sns.boxplot(data=df, x="region", y="price", ax=axes[1, 0])
axes[1, 0].set_title("Price Distribution by Region")

# Heatmap (correlation matrix)
sns.heatmap(df[["price", "quantity"]].corr(), annot=True, ax=axes[1, 1])
axes[1, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.show()

# Plotly - Interactive visualization
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter with hover data
fig = px.scatter(
    df,
    x="price",
    y="quantity",
    color="region",
    size="total_amount",
    hover_data=["customer_id", "date"],
    title="Interactive Price vs Quantity Analysis"
)
fig.update_layout(template="plotly_white")
fig.show()

# Time series with range slider
fig = px.line(
    df.group_by("date").agg(pl.col("price").sum()).to_pandas(),
    x="date",
    y="price",
    title="Revenue Over Time"
)
fig.update_xaxes(rangeslider_visible=True)
fig.show()
```

## Task Templates

### Task Template: Creating New Notebooks

**When to use**: Starting a new episode notebook from scratch

**Step-by-step process**:

1. **Setup Phase**:
   ```bash
   # Navigate to season directory
   cd saturdata/resources/season-1/

   # Create episode directory
   mkdir new-topic-name

   # Create notebook file
   touch new-topic-name/resource.py
   ```

2. **Notebook Scaffolding**:
   ```python
   import marimo

   __generated_with = "0.18.0"
   app = marimo.App(width="medium")


   @app.cell
   def _(mo):
       # Opening hook - episode connection
       mo.md(r"""
       # Episode Title

       **Episode XX**: Brief podcast reference

       ## What You'll Discover
       - Key learning objective 1
       - Key learning objective 2
       - Key learning objective 3
       """)
       return


   @app.cell
   def _():
       # Imports
       import marimo as mo
       import polars as pl
       import duckdb
       from pathlib import Path
       return duckdb, mo, Path, pl


   @app.cell
   def _(Path):
       # Path setup
       NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
       DATA_DIR = NOTEBOOK_DIR.parent / "data"
       return DATA_DIR, NOTEBOOK_DIR


   # Add progressive complexity cells here
   # Level 1: "Just Looking"
   # Level 2: "Getting Hands Dirty"
   # Level 3: "Challenge Accepted"


   if __name__ == "__main__":
       app.run()
   ```

3. **Progressive Difficulty Implementation**:
   - **Level 1**: Pre-written code with extensive inline comments
   - **Level 2**: Partially completed with TODO sections and hints
   - **Level 3**: Open-ended challenge with multiple valid approaches

4. **Testing Checklist**:
   - [ ] All cells execute without errors
   - [ ] Data loads correctly with absolute paths
   - [ ] Visualizations render properly
   - [ ] Markdown cells display formatted content
   - [ ] Episode references are contextual, not dependent on listening
   - [ ] Community engagement CTAs included (#Saturdata)

**Code scaffolding for progressive levels**:

```python
@app.cell
def _(mo):
    mo.md("""
    ## Level 1: "Just Looking"

    Run this cell to see the concept in action. No coding required!
    """)
    return


@app.cell
def _(df):
    # Fully implemented example with detailed comments
    # Users can modify parameters to explore
    result = df.group_by("region").agg([
        pl.col("price").mean().alias("avg_price"),  # Average price per region
        pl.col("quantity").sum().alias("total_quantity")  # Total items sold
    ])
    return (result,)


@app.cell
def _(mo):
    mo.md("""
    ## Level 2: "Getting Hands Dirty"

    Complete the TODOs below to apply the concept yourself.
    """)
    return


@app.cell
def _(df):
    # TODO: Group by product_id instead of region
    # TODO: Calculate median price instead of mean
    # Hint: Use .median() instead of .mean()

    result = df.group_by("___").agg([  # Fill in the blank
        pl.col("price").___().alias("median_price"),  # Choose aggregation
    ])
    return (result,)


@app.cell
def _(mo):
    mo.md("""
    ## Level 3: "Challenge Accepted"

    **Your Challenge**: Analyze customer purchasing patterns across regions and identify
    the top 3 products per region by revenue.

    **Multiple Approaches Welcome**: Use SQL, Polars, or Pandas - choose what feels natural!
    """)
    return


@app.cell
def _(df):
    # Your solution here
    # Share your approach on LinkedIn with #SaturdataChallenge
    pass
```

**Common pitfalls**:
- Forgetting absolute paths for data loading
- Not including episode context for non-listeners
- Skipping community engagement elements
- Making assumptions about prior knowledge

### Task Template: Improving Existing Notebooks

**When to use**: Enhancing accuracy, fixing bugs, or adding features to existing notebooks

**Step-by-step process**:

1. **Discovery Phase**:
   ```python
   # Read entire notebook first
   # Understand:
   # - Educational structure and flow
   # - Episode context and references
   # - Technical patterns used
   # - Data dependencies
   ```

2. **Verification Using Context7 MCP**:
   ```python
   # For DuckDB accuracy verification:
   # 1. Use Context7 to resolve library ID
   mcp__context7__resolve-library-id(libraryName="DuckDB")

   # 2. Get official documentation
   mcp__context7__get-library-docs(
       context7CompatibleLibraryID="/websites/duckdb-stable",
       topic="PIVOT UNPIVOT syntax",
       mode="code"
   )

   # 3. Compare notebook syntax with official examples
   # 4. Update to native DuckDB patterns when available
   ```

3. **Pattern Preservation Guidelines**:
   - Maintain existing educational structure (don't change level progression)
   - Preserve conversational tone and host references
   - Keep community engagement CTAs
   - Respect existing data loading patterns
   - Update only technical inaccuracies or add missing features

4. **Quality Assurance Checks**:
   - [ ] All code executes successfully
   - [ ] Technical accuracy verified against official docs
   - [ ] Educational flow maintained
   - [ ] Conversational tone preserved
   - [ ] Data paths still use absolute references
   - [ ] No regressions in existing functionality
   - [ ] Documentation updated if patterns changed

**Common corrections needed**:
- PostgreSQL references → DuckDB (for SQL notebook)
- `PERCENTILE_CONT() WITHIN GROUP` → `quantile_cont()` (native DuckDB)
- Relative paths → Absolute paths
- Missing episode context → Add natural references
- Technical jargon → Plain English explanations

### Task Template: Working with Data

**When to use**: Adding new datasets, generating synthetic data, or modifying data loading

**Step-by-step process**:

1. **Data Discovery Workflow**:
   ```python
   # Check if data exists in shared season-1/data/ directory
   from pathlib import Path

   DATA_DIR = Path("/Users/samlafell/Documents/saturdata/resources/season-1/data")

   # List available datasets
   print("Shared datasets:")
   for file in DATA_DIR.rglob("*.csv"):
       print(f"  - {file.relative_to(DATA_DIR)}")

   for file in DATA_DIR.rglob("*.parquet"):
       print(f"  - {file.relative_to(DATA_DIR)}")
   ```

2. **Path Setup Procedure**:
   ```python
   # Always use this pattern in notebooks
   NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
   DATA_DIR = NOTEBOOK_DIR.parent / "data"

   # Shared data
   transactions_path = DATA_DIR / "transactions_synthetic.csv"

   # Episode-specific data (if needed)
   episode_data_dir = NOTEBOOK_DIR / "data"
   customers_path = episode_data_dir / "customers.csv"
   ```

3. **Schema Documentation Pattern**:
   ```python
   # Load and document dataset
   transactions = pl.read_csv(str(DATA_DIR / "transactions_synthetic.csv"))

   mo.md(f"""
   **Dataset**: Synthetic E-Commerce Transactions

   **Schema**:
   ```
   {transactions.describe()}
   ```

   **Columns**:
   - `date` (Date): Transaction date (2024-01-01 to 2024-12-31)
   - `customer_id` (String): Unique customer identifier
   - `product_id` (String): Unique product identifier
   - `quantity` (Int32): Number of items purchased
   - `price` (Float64): Unit price in USD
   - `region` (String): Geographic region (North, South, East, West)
   - `promo_code` (String): Applied promotion code or NULL

   **Records**: {transactions.height:,}
   **Date Range**: {transactions["date"].min()} to {transactions["date"].max()}
   **Unique Customers**: {transactions["customer_id"].n_unique():,}
   """)
   ```

4. **Validation Approach**:
   ```python
   # Validate data loading
   assert transactions.height > 0, "No data loaded"
   assert "date" in transactions.columns, "Missing 'date' column"
   assert transactions["date"].null_count() == 0, "NULL dates found"

   # Type validations
   assert transactions["price"].dtype == pl.Float64, "Price should be Float64"
   assert transactions["quantity"].dtype == pl.Int32, "Quantity should be Int32"

   # Business rule validations
   assert transactions["price"].min() > 0, "Negative prices found"
   assert transactions["quantity"].min() > 0, "Negative quantities found"
   ```

**Common pitfalls**:
- Using relative paths (breaks in different environments)
- Not documenting schema and purpose
- Skipping validation checks
- Forgetting type conversions (especially dates)

### Task Template: DuckDB SQL Accuracy

**When to use**: Verifying or correcting DuckDB SQL syntax in notebooks

**Context7 Verification Workflow**:

1. **Resolve DuckDB Library**:
   ```
   Use: mcp__context7__resolve-library-id(libraryName="DuckDB")
   Select: /websites/duckdb-stable (High reputation, 2679 snippets)
   ```

2. **Get Documentation for Specific Features**:
   ```
   For PIVOT/UNPIVOT:
   mcp__context7__get-library-docs(
       context7CompatibleLibraryID="/websites/duckdb-stable",
       topic="PIVOT UNPIVOT syntax",
       mode="code"
   )

   For window functions:
   mcp__context7__get-library-docs(
       context7CompatibleLibraryID="/websites/duckdb-stable",
       topic="window functions QUALIFY FILTER",
       mode="code"
   )

   For statistical functions:
   mcp__context7__get-library-docs(
       context7CompatibleLibraryID="/websites/duckdb-stable",
       topic="PERCENTILE_CONT quantile_cont median",
       mode="code"
   )
   ```

**Native DuckDB Syntax Preferences**:

```sql
-- ✅ PREFERRED: Native DuckDB functions
SELECT
    region,
    median(amount) as median_amount,  -- Not PERCENTILE_CONT
    quantile_cont(amount, 0.25) as q1_amount,  -- For quartiles
    quantile_cont(amount, 0.75) as q3_amount
FROM transactions
GROUP BY region;

-- ❌ AVOID: SQL Standard syntax (while supported, less idiomatic)
SELECT
    region,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_amount
FROM transactions
GROUP BY region;

-- ✅ PREFERRED: Native DuckDB PIVOT
PIVOT monthly_sales
ON region
USING sum(sales)
GROUP BY month;

-- ✅ PREFERRED: Native DuckDB UNPIVOT
UNPIVOT pivoted_data
ON north_sales, south_sales, east_sales, west_sales
INTO
    NAME region
    VALUE sales;

-- ✅ Also supported: SQL Standard UNPIVOT
FROM pivoted_data
UNPIVOT (
    sales
    FOR region IN (north_sales, south_sales, east_sales, west_sales)
);
```

**Common Corrections Patterns**:

| Incorrect | Correct | Reason |
|-----------|---------|--------|
| `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` | `median(x)` | Native DuckDB function |
| `PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY x)` | `quantile_cont(x, p)` | Native syntax, cleaner |
| Missing `GROUP BY` in PIVOT | Add `GROUP BY month` | Required in DuckDB PIVOT |
| PostgreSQL references | DuckDB references | Wrong database system |

**Documentation Examples to Add**:

```python
mo.md("""
**🎯 DuckDB Statistical Functions Explained:**
- **median(amount)**: Native DuckDB function for median calculation
- **quantile_cont(amount, p)**: Continuous quantile/percentile function
    - p=0.25 → Q1 (first quartile)
    - p=0.50 → Median (same as median() function)
    - p=0.75 → Q3 (third quartile)
    - p=0.01/0.99 → Extreme percentiles for outlier detection

**Alternative Approaches:**
- DuckDB also supports SQL standard `PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY amount)`
- For multiple quantiles: `quantile_cont(amount, [0.25, 0.5, 0.75])` returns a list
""")
```

## Quality Assurance Checklist

Before marking any notebook work as complete, verify:

### Code Execution
- [ ] All cells execute without errors in fresh session
- [ ] Data loads successfully with absolute paths
- [ ] No hardcoded paths (use `NOTEBOOK_DIR`, `DATA_DIR` pattern)
- [ ] Type conversions work correctly (especially dates)
- [ ] No dependencies beyond `pyproject.toml`

### Educational Value
- [ ] Can be understood without listening to episode
- [ ] Progressive difficulty structure (3 levels)
- [ ] Includes at least one "aha moment" not in episode
- [ ] References real workplace scenarios
- [ ] Takes 15-30 minutes to complete fully
- [ ] Conversational tone maintained throughout

### Technical Accuracy
- [ ] Verified against official documentation (use Context7 for DuckDB)
- [ ] Native library functions used where available
- [ ] No deprecated or non-standard syntax
- [ ] Performance patterns follow best practices

### Community Engagement
- [ ] #Saturdata hashtag included
- [ ] Social sharing prompts present
- [ ] LinkedIn/GitHub formatting provided
- [ ] Challenge problems encourage sharing

### Visual Elements
- [ ] Plots and charts render correctly
- [ ] Interactive tables display properly
- [ ] Markdown formatting clean and readable
- [ ] Code syntax highlighting works

## Performance Optimization Patterns

### Expected Performance Ratios (Benchmarking Reference)

From polars-pandas notebook on Apple M1 (16GB RAM):

| Library | Time (seconds) | Speedup | Memory (MB) | Use Case |
|---------|----------------|---------|-------------|----------|
| Pandas Classic | 45-60s | 1x (baseline) | ~800MB | Legacy codebases |
| Pandas + PyArrow | 18-25s | 2.4x | ~400MB | Pandas with columnar backend |
| Polars | 2-3s | 23x | ~200MB | Modern ETL, large datasets |
| DuckDB | 2-4s | 21x | ~180MB | SQL analytics, in-memory queries |

### Optimization Techniques

```sql
-- ✅ GOOD: Filter early (WHERE before GROUP BY)
SELECT region, COUNT(*) as count
FROM transactions
WHERE date >= '2024-01-01'  -- Reduce rows before aggregation
GROUP BY region;

-- ❌ BAD: Filter late
SELECT region, COUNT(*) as count
FROM transactions
GROUP BY region
HAVING MIN(date) >= '2024-01-01';  -- Processes all rows first

-- ✅ GOOD: Use HAVING for post-aggregation filters
SELECT customer_id, COUNT(*) as order_count
FROM transactions
WHERE date >= '2024-01-01'  -- Filter rows first
GROUP BY customer_id
HAVING COUNT(*) >= 10;  -- Filter aggregated results

-- ✅ GOOD: Index-friendly conditions
WHERE date >= '2024-01-01' AND date < '2025-01-01'

-- ❌ BAD: Function on column prevents index usage
WHERE EXTRACT(YEAR FROM date) = 2024
```

### Query Plan Analysis

```python
# Use EXPLAIN to understand query execution
plan = mo.sql(f"""
    EXPLAIN
    SELECT region, SUM(price * quantity) as revenue
    FROM transactions
    WHERE date >= '2024-01-01'
    GROUP BY region
""")

# Look for:
# - Sequential scans vs index scans
# - Hash aggregates (fast) vs sort aggregates
# - Filter pushdown (good) vs late filtering
```

## DuckDB-Specific Guidelines

### Version Compatibility
- **Minimum**: DuckDB >= 1.4.1
- **Native PIVOT/UNPIVOT**: Supported since 0.8.0
- **QUALIFY clause**: DuckDB-specific feature
- **FILTER clause**: Standard SQL with DuckDB optimizations

### Native Functions Reference

```sql
-- Statistical functions
median(column)                          -- Median value
quantile_cont(column, 0.25)            -- Continuous quantile (Q1)
quantile_cont(column, [0.25, 0.5, 0.75])  -- Multiple quantiles (returns list)
approx_quantile(column, 0.5)           -- Approximate median (faster)

-- Aggregate functions
first(column ORDER BY date)            -- First value with ordering
last(column ORDER BY date)             -- Last value with ordering
list(column ORDER BY value)            -- Aggregate to list with ordering
string_agg(column, ', ' ORDER BY col)  -- String concatenation

-- Window functions
ROW_NUMBER() OVER (...)                -- Sequential numbering
RANK() OVER (...)                      -- Ranking with gaps
DENSE_RANK() OVER (...)                -- Ranking without gaps
NTILE(n) OVER (...)                    -- Divide into n buckets
LAG(column, offset) OVER (...)         -- Previous row value
LEAD(column, offset) OVER (...)        -- Next row value

-- DuckDB-specific clauses
QUALIFY ROW_NUMBER() OVER (...) <= 10  -- Filter window function results
COUNT(*) FILTER (WHERE condition)      -- Conditional aggregation
```

### PIVOT/UNPIVOT Examples

```sql
-- PIVOT: Convert rows to columns
PIVOT monthly_regional_sales
ON region                               -- Column to pivot on
USING sum(sales)                        -- Aggregation function
GROUP BY month;                         -- Grouping columns

-- Result: month | north_sales | south_sales | east_sales | west_sales

-- UNPIVOT: Convert columns to rows
UNPIVOT pivoted_data
ON north_sales, south_sales, east_sales, west_sales
INTO
    NAME region                         -- New category column
    VALUE sales;                        -- New value column

-- Result: month | region | sales

-- SQL Standard UNPIVOT (also supported)
FROM pivoted_data
UNPIVOT (
    sales                               -- Value column name
    FOR region IN (north_sales, south_sales, east_sales, west_sales)
);
```

### Common DuckDB Pitfalls

| Pitfall | Solution | Example |
|---------|----------|---------|
| Forgetting GROUP BY in PIVOT | Always specify grouping columns | `PIVOT ... GROUP BY month` |
| Using SQL Standard syntax | Prefer native DuckDB functions | `median(x)` not `PERCENTILE_CONT` |
| Not using QUALIFY | Filter window functions directly | `QUALIFY ROW_NUMBER() ... <= 10` |
| Ignoring FILTER clause | Use for conditional aggregations | `COUNT(*) FILTER (WHERE ...)` |

## References

- **Educational Guidelines**: See `style.md` for content structure, tone, and community engagement patterns
- **Official Docs**: Use Context7 MCP with `/websites/duckdb-stable` for DuckDB accuracy
- **Marimo Docs**: https://docs.marimo.io/ for notebook features and best practices
- **Polars Docs**: https://pola-rs.github.io/polars/ for DataFrame operations
