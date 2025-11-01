# Data Mechanic's Garage: Python Data Processing Performance

Welcome to Episode 1 of Saturdata Season 1! This interactive learning environment explores modern Python data processing libraries through hands-on performance benchmarking.

## 📚 What You'll Learn

### Core Topics Covered
- **Pandas Baseline**: Understanding traditional DataFrame operations and their limitations
- **PyArrow Integration**: Quick performance wins with minimal code changes
- **Polars**: Modern parallel processing engine with lazy evaluation
- **DuckDB**: SQL interface for in-memory analytics
- **Performance Benchmarking**: Timing and memory profiling techniques
- **Decision Framework**: When to use each library in production

### Key Concepts
1. **Performance Optimization**: 20-30x speedups over baseline Pandas
2. **Memory Efficiency**: Columnar storage and optimization techniques
3. **API Design**: Expression-based vs procedural approaches
4. **Lazy Evaluation**: Query optimization before execution
5. **Parallel Processing**: Multi-core CPU utilization
6. **Cloud Migration Paths**: Local tools → Cloud platforms (DuckDB→Snowflake, Polars→Databricks)

### Learning Materials
1. **`resource.py`**: Interactive Marimo notebook with executable benchmarks
2. **`overview.md`**: Comprehensive reference guide with theory and examples
3. **`README.md`**: This setup and usage guide

## 🗂 Dataset

### Synthetic E-Commerce Data
Generated realistic transaction data for performance testing:

#### Transactions (`data/transactions.csv`)
- **5 million rows** (~1GB CSV file)
- Transaction date, customer ID, product ID, quantity, price, revenue
- Product categories, regional distribution, promotional codes

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | INTEGER | Unique transaction identifier |
| `date` | DATE | Transaction date (2023-2024) |
| `customer_id` | INTEGER | Customer identifier (1-100,000) |
| `product_id` | INTEGER | Product identifier (1-10,000) |
| `product_category` | TEXT | Category (Electronics, Clothing, etc.) |
| `quantity` | INTEGER | Quantity purchased (1-10) |
| `price` | DECIMAL | Unit price ($5-500) |
| `region` | TEXT | Geographic region (North, South, East, West) |
| `promo_code` | TEXT | Promotional code (PROMO10/20/30 or NULL) |
| `revenue` | DECIMAL | Total revenue after discounts |

#### Customers (`data/customers.csv`)
- **100,000 rows** - Customer dimension table
- Customer ID, email, segment, signup date, lifetime value

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | INTEGER | Unique customer identifier |
| `email` | TEXT | Customer email address |
| `segment` | TEXT | Customer segment (Premium, Standard, Basic) |
| `signup_date` | TIMESTAMP | Account creation date |
| `lifetime_value` | DECIMAL | Total customer value |

#### Products (`data/products.csv`)
- **10,000 rows** - Product dimension table
- Product ID, name, category, cost, weight, supplier

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | INTEGER | Unique product identifier |
| `product_name` | TEXT | Product name |
| `category` | TEXT | Product category |
| `cost` | DECIMAL | Product cost |
| `weight_kg` | DECIMAL | Shipping weight |
| `supplier_id` | INTEGER | Supplier identifier |

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.11+** (required for optimal performance)
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/))
- **8GB+ RAM** recommended for 5M row benchmarks
- **Basic Python knowledge**: Functions, loops, data structures

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   gh repo clone saturdata/resources
   cd resources/season-1/polars_pandas
   ```

2. **Generate the dataset** (one-time setup):
   ```bash
   python data/generate_dataset.py
   ```

   This creates three CSV files in `data/`:
   - `transactions.csv` (5M rows, ~330MB)
   - `customers.csv` (100K rows, ~6.6MB)
   - `products.csv` (10K rows, ~444KB)

3. **Launch the notebook**:
   ```bash
   uv run marimo run resource.py
   ```

   **Important**: Always run from the `polars_pandas/` directory so paths resolve correctly.

   The `uv run` command will automatically:
   - Create a virtual environment
   - Install dependencies: `marimo`, `pandas`, `polars`, `duckdb`, `pyarrow`, `altair`
   - Launch the interactive notebook in your browser

### Verify Installation

If you encounter issues, verify your environment:

```bash
# Check Python version (should be 3.11+)
python --version

# Check uv installation
uv --version

# Verify data files exist
ls data/transactions.csv
ls data/customers.csv
ls data/products.csv
```

## 🎯 Getting Started

### Option 1: Interactive Learning (Recommended)
Start the Marimo notebook for hands-on benchmarking:

```bash
uv run marimo run resource.py
```

The notebook provides:
- **Executable code cells** with real-time performance metrics
- **Interactive visualizations** comparing library performance
- **Progressive complexity** from baseline Pandas to modern tools
- **Detailed explanations** of why performance differs

**Recommended workflow:**
1. Read the introduction and understand the benchmark setup
2. Run each library benchmark cell and observe timing/memory
3. Study the code patterns and API differences
4. Experiment with modifications and re-run benchmarks
5. Review the decision framework for production use

### Option 2: Reference Study
Use `overview.md` as a comprehensive reference:
- Detailed explanations of each library's architecture
- Performance characteristics and trade-offs
- Code examples with annotations
- Migration guides (Pandas → Polars, SQL → DuckDB)
- Production deployment best practices

### Option 3: Quick Start Guide
For experienced users who want quick insights:

1. **Run all benchmarks**: Execute the notebook start to finish
2. **Review scoreboard**: See the final performance comparison table
3. **Read decision framework**: Understand when to use each tool
4. **Try one migration**: Rewrite a Pandas script in Polars

## 📊 Benchmark Operations

The notebook tests realistic data workflows:

### 1. Data Loading
- Read 5M rows from CSV
- Parse data types
- Measure I/O performance

### 2. String Manipulation
- Extract email domains from customer data
- Test vectorized operations vs `.apply()`
- Compare expression-based APIs

### 3. Aggregations
- Group by region and product category
- Multiple aggregation functions (sum, mean, count, nunique)
- Test parallel execution capabilities

### 4. Joins
- Left join transactions with customer dimension (5M × 100K rows)
- Test join optimization strategies
- Measure memory efficiency

## 🔧 Expected Results

Based on our testing (Apple M1, 16GB RAM):

| Library | Time (seconds) | Memory (MB) | Speedup |
|---------|----------------|-------------|---------|
| Classic Pandas | 45-60s | ~2,400 MB | 1.0x (baseline) |
| Pandas + PyArrow | 18-25s | ~1,800 MB | 2.4x |
| **Polars** 🚀 | **2-3s** | **~600 MB** | **23x** |
| **DuckDB** 🐥 | **2-4s** | **~450 MB** | **21x** |

*Note: Results vary based on CPU cores, RAM, and disk speed.*

## 🎓 Learning Path

### Week 1: Quick Wins
- ✅ Run all benchmarks and understand baseline performance
- ✅ Add PyArrow backend to your existing Pandas code
- ✅ Measure improvements in your real workflows
- ✅ Identify performance bottlenecks

### Week 2-3: Learn Modern Tools
- ✅ Study Polars expression API and lazy evaluation
- ✅ Practice DuckDB SQL queries on sample data
- ✅ Rewrite one slow Pandas script in Polars
- ✅ Compare code readability and performance

### Month 2: Production Migration
- ✅ Migrate critical data pipelines to Polars/DuckDB
- ✅ Set up performance monitoring and alerting
- ✅ Document decision framework for your team
- ✅ Train colleagues on new tools

### Month 3+: Advanced Patterns
- ✅ Master streaming for larger-than-memory datasets
- ✅ Optimize query plans with lazy evaluation
- ✅ Explore cloud deployment (Snowflake, Databricks)
- ✅ Implement data quality checks and testing

## 🏗️ Project Structure

```
polars_pandas/
├── resource.py              # Interactive Marimo notebook
├── README.md               # This file (setup guide)
├── overview.md             # Comprehensive reference
└── data/
    ├── generate_dataset.py  # Dataset generation script
    ├── transactions.csv     # 5M transaction records
    ├── customers.csv        # 100K customer records
    └── products.csv         # 10K product records
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'marimo'"
**Solution**: Ensure you're using `uv run` prefix:
```bash
uv run marimo run resource.py
```

#### "FileNotFoundError: data/transactions.csv"
**Cause**: Either the dataset hasn't been generated, or you're running marimo from the wrong directory.

**Solution 1 - Generate the dataset**:
```bash
# Make sure you're in polars_pandas/ directory
pwd  # Should show .../season-1/polars_pandas
python data/generate_dataset.py
```

**Solution 2 - Run from correct directory**:
```bash
# Navigate to the polars_pandas directory
cd /path/to/resources/season-1/polars_pandas
uv run marimo run resource.py
```

The notebook uses paths relative to its location, so it must be run from the `polars_pandas/` directory.

#### "Memory Error" during benchmarks
**Solution**: Reduce dataset size by modifying `generate_dataset.py`:
```python
transactions = generate_transactions(1_000_000)  # Instead of 5M
```

#### Slow performance on Windows
**Solution**:
- Ensure you're using Python 3.11+ (significant performance improvements)
- Close other memory-intensive applications
- Consider using WSL2 for better I/O performance

#### PyArrow compatibility issues
**Solution**: Ensure compatible versions:
```bash
uv pip install "pandas>=2.0.0" "pyarrow>=12.0.0"
```

## 🔗 Additional Resources

### Official Documentation
- [Marimo](https://docs.marimo.io/) - Reactive Python notebooks
- [Polars User Guide](https://pola-rs.github.io/polars-book/) - Comprehensive Polars documentation
- [DuckDB Documentation](https://duckdb.org/docs/) - SQL reference and guides
- [Pandas PyArrow](https://pandas.pydata.org/docs/user_guide/pyarrow.html) - Integration guide

### Performance Benchmarks
- [DuckDB Benchmark Suite](https://duckdblabs.github.io/db-benchmark/) - Independent benchmarks
- [Polars Benchmarks](https://www.pola.rs/posts/benchmarks/) - Official performance comparisons

### Community & Support
- [Polars Discord](https://discord.gg/4UfP5cfBE7) - Active community support
- [DuckDB Discussions](https://github.com/duckdb/duckdb/discussions) - GitHub discussions
- [r/dataengineering](https://reddit.com/r/dataengineering) - Reddit community

### Blog Posts & Tutorials
- "Pandas 2.0 and PyArrow: Better, Faster, Stronger" - Official Pandas blog
- "Why Polars uses 5x less memory" - Polars blog
- "DuckDB: The SQLite for Analytics" - DuckDB blog

## 💡 Tips for Success

### Performance Optimization
1. **Profile first**: Use benchmarking to identify actual bottlenecks
2. **Start simple**: Try PyArrow backend before full rewrites
3. **Measure impact**: Track time and memory improvements
4. **Document decisions**: Record why you chose each tool

### Code Quality
1. **Type hints**: Add type annotations for better IDE support
2. **Consistent style**: Follow your team's conventions
3. **Clear names**: Use descriptive variable names
4. **Comments**: Explain *why*, not *what*

### Production Deployment
1. **Test thoroughly**: Verify results match Pandas output
2. **Monitor performance**: Set up alerts for regressions
3. **Version control**: Pin exact library versions
4. **Document**: Create runbooks for operations team

## 🎯 Next Steps

After completing this episode:

1. **Experiment**: Modify the benchmarks with your own data patterns
2. **Apply**: Rewrite one slow script from your work using Polars
3. **Share**: Show your team the performance improvements
4. **Explore**: Check out the next Saturdata episode on data visualization

## 🤝 Contributing

Found an issue or want to improve the notebook?

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This educational content is provided under the MIT License. See the main repository for details.

---

**Ready to tune up your data processing?** 🔧

```bash
uv run marimo run resource.py
```

Happy coding! 🚀
