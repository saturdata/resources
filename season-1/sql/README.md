# SQL for Data People - Learning Resources

Welcome to the comprehensive SQL learning environment designed for data scientists, analysts, and engineers. This repository provides both theoretical knowledge and hands-on practice with advanced PostgreSQL concepts using real-world datasets.

## 📚 What You'll Learn

### Core Topics Covered
- **Window Functions**: ROW_NUMBER(), RANK(), LAG/LEAD, running totals, moving averages
- **Advanced PostgreSQL Commands**: QUALIFY, FILTER, WITHIN GROUP, array/JSON operations
- **CTE vs Subquery Framework**: Performance trade-offs and decision guidelines
- **PIVOT/UNPIVOT Operations**: Data transformation techniques without native operators
- **Advanced SQL Patterns**: Gap/island analysis, deduplication, time series patterns
- **Idempotent Operations**: UPSERT patterns, incremental updates, data synchronization
- **Performance Optimization**: Query analysis, debugging, and optimization strategies

### Learning Materials
1. **`overview.md`**: Comprehensive reference guide with theory and examples
2. **`resource.py`**: Interactive Marimo notebook with executable SQL examples
3. **`README.md`**: This setup and usage guide

## 🗂 Data Sources

### Transaction Data (`../data/transactions_synthetic.csv`)
Synthetic e-commerce transaction data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | DATE | Transaction date (2024 data) |
| `customer_id` | INTEGER | Unique customer identifier |
| `product_id` | INTEGER | Unique product identifier |
| `price` | DECIMAL | Unit price of the product |
| `quantity` | INTEGER | Quantity purchased |
| `region` | TEXT | Geographic region (North, South, East, West) |
| `promo_code` | TEXT | Promotional code used (nullable) |

**Business Context:**
- ~50,000 transactions across 2024
- ~25,000 unique customers
- ~500 unique products  
- 4 geographic regions
- 3 promotional codes (PROMO10, PROMO20, PROMO30)

### NYC Taxi Data (`../data/tlc/*.parquet`)
Real NYC Yellow Taxi trip data for January and February 2024:

| Column | Type | Description |
|--------|------|-------------|
| `tpep_pickup_datetime` | TIMESTAMP | Trip start time |
| `tpep_dropoff_datetime` | TIMESTAMP | Trip end time |
| `passenger_count` | INTEGER | Number of passengers |
| `trip_distance` | FLOAT | Trip distance in miles |
| `fare_amount` | FLOAT | Base fare amount |
| `tip_amount` | FLOAT | Tip amount |
| `total_amount` | FLOAT | Total charge to customer |
| `PULocationID` | INTEGER | Pickup location zone |
| `DOLocationID` | INTEGER | Drop-off location zone |

**Business Context:**
- ~3 million trips across Jan-Feb 2024
- Real temporal and geographic patterns
- Multiple pricing components
- Rich data for time series analysis

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.8+** 
- **Basic SQL Knowledge**: Understanding of SELECT, JOIN, GROUP BY, WHERE clauses
- **Optional**: PostgreSQL familiarity (concepts transfer to DuckDB)

### Required Packages
Install the following Python packages:

```bash
pip install marimo duckdb pandas pyarrow
```

**Package Details:**
- **`marimo`**: Interactive notebook environment for data exploration
- **`duckdb`**: In-memory analytical database (no external setup required)
- **`pandas`**: Data manipulation and analysis
- **`pyarrow`**: Parquet file support for taxi data

### Environment Setup

1. **Clone/Download**: Ensure you have access to this repository structure:
   ```
   season-1/
   ├── data/                    # Data files
   │   ├── transactions_synthetic.csv
   │   └── tlc/
   │       ├── yellow_tripdata_2024-01.parquet
   │       └── yellow_tripdata_2024-02.parquet
   └── sql/                     # SQL learning materials
       ├── overview.md              # Reference guide
       ├── resource.py              # Interactive notebook
       └── README.md               # This file
   ```

2. **Verify Data Access**: Ensure the data files are accessible from the `sql/` directory:
   ```bash
   ls ../data/transactions_synthetic.csv
   ls ../data/tlc/*.parquet
   ```

3. **Test Environment**: Quick verification:
   ```python
   import duckdb
   import pandas as pd
   import pyarrow.parquet as pq
   
   # Test DuckDB
   conn = duckdb.connect(':memory:')
   result = conn.execute("SELECT 'Setup successful!' as status").fetchone()
   print(result[0])  # Should print: Setup successful!
   ```

## 🎯 Getting Started

### Option 1: Interactive Learning (Recommended)
Start with the Marimo notebook for hands-on learning:

```bash
cd season-1/sql/
marimo run resource.py
```

This will open an interactive notebook in your browser where you can:
- Execute SQL queries against real data
- Modify examples to experiment with concepts
- See immediate results and explanations
- Progress through topics at your own pace

### Option 2: Reference Study
Use `overview.md` as a comprehensive reference guide:
- Detailed explanations of each SQL concept
- Code examples with business context
- Performance tips and best practices
- Decision frameworks for choosing techniques

### Option 3: Combined Approach
1. Read the relevant section in `overview.md`
2. Run the corresponding examples in `resource.py`
3. Experiment with variations and parameters
4. Apply concepts to your own datasets
