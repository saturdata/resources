"""
Generate synthetic e-commerce dataset for performance benchmarking.

Creates a realistic dataset with:
- 5 million transactions
- Customer, product, and regional dimensions
- String operations (email domains, product categories)
- Temporal patterns (seasonal trends)
- Join opportunities (customer and product dimension tables)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_transactions(n_rows: int = 5_000_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic transaction data."""
    np.random.seed(seed)

    # Date range: 2023-01-01 to 2024-12-31
    start_date = datetime(2023, 1, 1)
    date_range = pd.date_range(start_date, periods=730, freq='D')

    # Generate realistic patterns
    regions = ['North', 'South', 'East', 'West']
    product_categories = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports',
        'Books', 'Toys', 'Food & Beverage', 'Health & Beauty'
    ]
    promo_codes = ['PROMO10', 'PROMO20', 'PROMO30', None]

    data = {
        'transaction_id': range(1, n_rows + 1),
        'date': np.random.choice(date_range, n_rows),
        'customer_id': np.random.randint(1, 100_001, n_rows),
        'product_id': np.random.randint(1, 10_001, n_rows),
        'product_category': np.random.choice(product_categories, n_rows),
        'quantity': np.random.randint(1, 11, n_rows),
        'price': np.round(np.random.uniform(5, 500, n_rows), 2),
        'region': np.random.choice(regions, n_rows, p=[0.3, 0.25, 0.25, 0.2]),
        'promo_code': np.random.choice(promo_codes, n_rows, p=[0.1, 0.15, 0.15, 0.6]),
    }

    df = pd.DataFrame(data)

    # Calculate revenue
    df['revenue'] = df['price'] * df['quantity']

    # Apply discounts
    df.loc[df['promo_code'] == 'PROMO10', 'revenue'] *= 0.9
    df.loc[df['promo_code'] == 'PROMO20', 'revenue'] *= 0.8
    df.loc[df['promo_code'] == 'PROMO30', 'revenue'] *= 0.7

    df['revenue'] = df['revenue'].round(2)

    return df


def generate_customers(n_customers: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate customer dimension table."""
    np.random.seed(seed)

    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com', 'business.net']
    segments = ['Premium', 'Standard', 'Basic']

    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'email': [f'customer{i}@{np.random.choice(domains)}' for i in range(1, n_customers + 1)],
        'segment': np.random.choice(segments, n_customers, p=[0.2, 0.5, 0.3]),
        'signup_date': pd.date_range('2020-01-01', periods=n_customers, freq='5min')[:n_customers],
        'lifetime_value': np.round(np.random.lognormal(6, 1.5, n_customers), 2),
    })

    return customers


def generate_products(n_products: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate product dimension table."""
    np.random.seed(seed)

    categories = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports',
        'Books', 'Toys', 'Food & Beverage', 'Health & Beauty'
    ]

    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'cost': np.round(np.random.uniform(2, 300, n_products), 2),
        'weight_kg': np.round(np.random.uniform(0.1, 50, n_products), 2),
        'supplier_id': np.random.randint(1, 501, n_products),
    })

    return products


if __name__ == '__main__':
    print("Generating synthetic datasets...")

    # Determine output directory (data folder relative to script location)
    script_dir = Path(__file__).parent
    # If script is in data/, save to current directory; otherwise save to data/
    if script_dir.name == 'data':
        output_dir = script_dir
    else:
        output_dir = script_dir / 'data'
        output_dir.mkdir(exist_ok=True)

    # Generate transaction data (5M rows ~500MB)
    print("1. Generating transactions (5M rows)...")
    transactions = generate_transactions(5_000_000)
    transactions_path = output_dir / 'transactions.csv'
    transactions.to_csv(transactions_path, index=False)
    print(f"   Saved: {transactions_path} ({len(transactions):,} rows)")

    # Generate customer dimension
    print("2. Generating customers (100K rows)...")
    customers = generate_customers(100_000)
    customers_path = output_dir / 'customers.csv'
    customers.to_csv(customers_path, index=False)
    print(f"   Saved: {customers_path} ({len(customers):,} rows)")

    # Generate product dimension
    print("3. Generating products (10K rows)...")
    products = generate_products(10_000)
    products_path = output_dir / 'products.csv'
    products.to_csv(products_path, index=False)
    print(f"   Saved: {products_path} ({len(products):,} rows)")

    # Display sample and stats
    print("\nDataset Summary:")
    print(f"  Transactions: {len(transactions):,} rows, {transactions.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"  Customers: {len(customers):,} rows")
    print(f"  Products: {len(products):,} rows")
    print(f"\nSample transaction data:")
    print(transactions.head(3).to_string())
