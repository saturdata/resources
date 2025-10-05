#!/usr/bin/env python3
"""
Analytic Basics in Python - Marimo Notebook
============================================
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
    # Analytics basics in Python
    Saturdata: Season 1

    This notebook covers fundamental analytics operations in Python including:
    - Statistics operations
    - Statistical tests
    - Tabular data operations
    - Data types in pandas and polars
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## TODO
    - Add `plotly` examples
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    # Import required libraries
    import time
    import sys
    import numpy as np
    import pandas as pd
    import polars as pl
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency, pearsonr
    import warnings

    warnings.filterwarnings("ignore")

    print("✅ All libraries imported successfully")
    return np, pd, pl, plt, sns, stats, time


@app.cell
def _(sns):
    # Set seaborn style for all charts with one quick shortcut
    sns.set()
    return


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
    mo.md(r"""## SciPy for statistical tests and distributions""")
    return


@app.cell
def _(np):
    # Generate sample data for statistical tests
    np.random.seed(42)
    group1 = np.random.normal(100, 10, 50)
    group2 = np.random.normal(105, 12, 50)
    return group1, group2


@app.cell
def _(mo):
    mo.md(r"""Note that random seeds allow you to reproduce the same random pattern for consistent operations in projects and testing""")
    return


@app.cell
def _(mo):
    mo.md(r"""`stats.ttest_ind` performs an independent samples t-test to determine if there's a statistically significant difference between the means of two independent groups (like comparing test scores between two different classes).""")
    return


@app.cell
def _(group1, group2, stats):
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"T-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""The negative t-statistic (-3.760) indicates that the first group has a significantly lower mean than the second group. With a p-value of essentially zero (< 0.001), we can reject the null hypothesis and conclude there is a statistically significant difference between the two groups.""")
    return


@app.cell
def _(mo):
    mo.md(r"""`stats.chi2_contingency` performs a chi-square test of independence to determine if there's a statistically significant association between two categorical variables in a contingency table (like testing if gender is related to product preference).""")
    return


@app.cell
def _(np, stats):
    observed = np.array([[10, 15, 5], [8, 12, 10]])
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)
    print(f"Chi-square test: χ² = {chi2:.3f}, p-value = {p_val:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""The chi-square test shows no significant association between the categorical variables being tested. With a p-value of 0.329 (well above the typical α = 0.05 threshold), we fail to reject the null hypothesis of independence. The observed frequencies in the contingency table are not significantly different from what we would expect by chance alone.""")
    return


@app.cell
def _(np, stats):
    # Normal distribution analysis
    normal_data = np.random.normal(50, 10, 1000)
    shapiro_stat, shapiro_p = stats.shapiro(normal_data[:100])  # Shapiro-Wilk test
    print(f"Shapiro-Wilk test for normality: W = {shapiro_stat:.3f}, p = {shapiro_p:.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The Shapiro-Wilk test assesses whether a dataset follows a normal (Gaussian) distribution by comparing the observed data to what would be expected from a perfect normal distribution. The test statistic W ranges from 0 to 1, where values closer to 1 indicate better normality.

    Our W = 0.978 suggests the data is very close to normal, and with p = 0.085 (above α = 0.05), we fail to reject the null hypothesis that the data is normally distributed. This means we can reasonably assume the dataset follows a normal distribution for subsequent statistical analyses that require this assumption.

    In the real world, we basically always assume normality and never check ML assumptions because these are the only tools we have, and your data almost always won't pass tests like this.
    """
    )
    return


@app.cell
def _(np, stats):
    # Correlation and regression with scipy
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = 2 * x + 1 + np.random.normal(0, 1, 10)
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
    print(f"Linear regression: slope={slope:.2f}, r²={r_value**2:.3f}, p={p_value_reg:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""The linear regression shows an excellent fit with a strong positive relationship. For every one-unit increase in the predictor variable, the response variable increases by approximately 2.05 units. The r² = 0.942 indicates that 94.2% of the variance in the dependent variable is explained by the linear relationship, demonstrating a very strong predictive model. The p-value of essentially zero confirms this relationship is highly statistically significant.""")
    return


@app.cell
def _(np, stats):
    # Distribution fitting
    sample_data = np.random.exponential(2, 1000)
    params = stats.expon.fit(sample_data)
    print(f"Exponential distribution parameters: {params}")
    return


@app.cell
def _(mo):
    mo.md(r"""The stats.expon.fit() function returns two parameters: location (loc) and scale. The first parameter (0.003) represents the location parameter (essentially the minimum value), which is very close to zero as expected for an exponential distribution. The second parameter (1.975) is the scale parameter, which closely matches our original exponential parameter of 2.0 used to generate the data. This confirms that scipy successfully identified the underlying exponential distribution with the correct rate parameter.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Seaborn visualizations

    Seaborn provides high-level, statistically-oriented plotting functions with beautiful default themes (see `sns.set()`) and color palettes that make creating publication-quality visualizations much faster than matplotlib's low-level API. It automatically handles pandas DataFrames, computes statistical aggregations (like confidence intervals), and produces complex multi-plot layouts with single function calls, whereas matplotlib would require dozens of lines of manual configuration to achieve the same results.

    Note: Seaborn works best with pandas. To get the best of both worlds, we recommend processing your data with Polars and then converting to pandas for viz.
    """
    )
    return


@app.cell
def _(np, pl):
    # Create sample dataset for visualization
    np.random.seed(42)
    viz_data = pl.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.exponential(2, 100)
    })
    return (viz_data,)


@app.cell
def _(viz_data):
    # Convert to pandas for seaborn (seaborn works best with pandas)
    viz_data_pd = viz_data.to_pandas()
    return (viz_data_pd,)


@app.cell
def _(plt, sns, viz_data_pd):
    # Scatter plot with seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=viz_data_pd, x='x', y='y', hue='category', size='value')
    plt.title('Scatter Plot with Categories and Sizes')
    plt.show()
    return


@app.cell
def _(plt, sns, viz_data_pd):
    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=viz_data_pd, x='value', hue='category', kde=True)
    plt.title('Distribution by Category')
    plt.show()
    return


@app.cell
def _(plt, sns, viz_data_pd):
    # Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=viz_data_pd, x='category', y='value')
    plt.title('Box Plot by Category')
    plt.show()
    return


@app.cell
def _(np, plt, sns, viz_data_pd):
    # Correlation heatmap
    correlation_data = viz_data_pd.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()
    return


@app.cell
def _(np, pl, plt, sns):
    # Pair plot for multiple variables
    sample_df = pl.DataFrame({
        'height': np.random.normal(170, 10, 100),
        'weight': np.random.normal(70, 15, 100),
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['M', 'F'], 100)
    }).to_pandas()

    sns.pairplot(sample_df, hue='gender')
    plt.suptitle('Pair Plot by Gender', y=1.02)
    plt.show()
    return


@app.cell
def _(sns):
    # Apply seaborn styling globally with sns.set()
    # This shortcut applies seaborn's default aesthetic settings to ALL matplotlib plots
    # including: figure style, context (sizing), color palette, and font scale
    # It affects both seaborn plots AND regular matplotlib plots created after this call
    sns.set()
    return


@app.cell
def _(plt, viz_data_pd):
    # Demonstrate how sns.set() affects regular matplotlib plots
    # Create a regular matplotlib chart using existing viz_data - notice the seaborn styling!
    fig_mpl, axes_mpl = plt.subplots(1, 2, figsize=(12, 5))

    # Pure matplotlib line chart - but with seaborn aesthetics applied
    x_data = viz_data_pd['x'].values
    y_data = viz_data_pd['y'].values
    axes_mpl[0].plot(x_data, y_data, 'o-', linewidth=2, markersize=6)
    axes_mpl[0].set_xlabel('X Values')
    axes_mpl[0].set_ylabel('Y Values')
    axes_mpl[0].set_title('Matplotlib Plot with Seaborn Theme')
    axes_mpl[0].grid(True, alpha=0.3)

    # Pure matplotlib bar chart - also styled by seaborn
    category_counts = viz_data_pd['category'].value_counts()
    axes_mpl[1].bar(category_counts.index, category_counts.values, alpha=0.8)
    axes_mpl[1].set_xlabel('Category')
    axes_mpl[1].set_ylabel('Count')
    axes_mpl[1].set_title('Matplotlib Bar Chart with Seaborn Theme')

    plt.tight_layout()
    plt.show()

    print("Notice: Regular matplotlib plots now have seaborn's clean aesthetic!")
    print("Benefits: Better colors, nicer grid, improved fonts, professional look")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Analytics combinations
    Let's combine these libraries to showcase a complete analysis
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
def _(business_data, pl, stats):
    # Regional analysis with statistical tests
    north_sales = business_data.filter(pl.col('region') == 'North')['sales'].to_numpy()
    south_sales = business_data.filter(pl.col('region') == 'South')['sales'].to_numpy()

    # Perform statistical test for significant different between north and south sales
    t_stat_ns, p_val_ns = stats.ttest_ind(north_sales, south_sales)
    print(f"\nNorth vs South sales t-test: t={t_stat_ns:.3f}, p={p_val_ns:.3f}")
    return


@app.cell
def _(business_data, plt, sns):
    # Visualization of the business analysis
    business_pd = business_data.to_pandas()

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Sales distribution by region
    sns.boxplot(data=business_pd, x='region', y='sales', ax=axes[0,0])
    axes[0,0].set_title('Sales by Region')

    # Marketing spend vs sales correlation
    sns.scatterplot(data=business_pd, x='marketing_spend', y='sales', 
                    hue='region', ax=axes[0,1])
    axes[0,1].set_title('Marketing Spend vs Sales')

    # Quarterly trends
    sns.barplot(data=business_pd, x='quarter', y='sales', ax=axes[1,0])
    axes[1,0].set_title('Sales by Quarter')

    # Distribution of sales
    sns.histplot(data=business_pd, x='sales', kde=True, ax=axes[1,1])
    axes[1,1].set_title('Sales Distribution')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Hypothesis testing""")
    return


@app.cell
def _(np):
    # Complete hypothesis testing example
    np.random.seed(42)

    # Simulate A/B test data
    control_group = np.random.normal(2.3, 0.5, 100)  # Control conversion rate
    treatment_group = np.random.normal(2.8, 0.6, 100)  # Treatment conversion rate
    return control_group, treatment_group


@app.cell
def _(control_group, np, treatment_group):
    # Descriptive statistics
    print("A/B Test Analysis:")
    print(f"Control: mean={np.mean(control_group):.3f}, std={np.std(control_group):.3f}")
    print(f"Treatment: mean={np.mean(treatment_group):.3f}, std={np.std(treatment_group):.3f}")
    return


@app.cell
def _(control_group, np, stats, treatment_group):
    # Statistical test
    t_stat_hyp, p_value_hyp = stats.ttest_ind(control_group, treatment_group)
    effect_size = (np.mean(treatment_group) - np.mean(control_group)) / np.sqrt(
        (np.var(control_group) + np.var(treatment_group)) / 2)

    print(f"T-test: t={t_stat_hyp:.3f}, p={p_value_hyp:.3f}")
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""The t-test shows a highly significant difference between the treatment and control groups (p < 0.001), with the treatment group performing significantly better. The Cohen's d of 1.100 indicates a large effect size (d > 0.8 is considered large), meaning the treatment produced a substantial practical difference beyond just statistical significance.""")
    return


@app.cell
def _(control_group, np, pl, plt, sns, treatment_group):
    # Visualization
    ab_data = pl.DataFrame({
        'group': ['Control']*100 + ['Treatment']*100,
        'value': np.concatenate([control_group, treatment_group])
    }).to_pandas()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ab_data, x='group', y='value')
    sns.stripplot(data=ab_data, x='group', y='value', alpha=0.5, size=3)
    plt.title('A/B Test Results')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Custom function
    Custom function that combines libraries
    """
    )
    return


@app.cell
def _(np, pl, plt, sns, stats):
    def comprehensive_analysis(data_df, numeric_col, category_col):
        """
        Perform comprehensive statistical analysis using all four libraries
        """
        # Convert to different formats as needed
        if isinstance(data_df, pl.DataFrame):
            pd_df = data_df.to_pandas()
            pl_df = data_df
        else:
            pd_df = data_df
            pl_df = pl.from_pandas(data_df)

        # NumPy statistics
        values = pl_df[numeric_col].to_numpy()
        np_stats = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'skew': stats.skew(values)
        }

        # Polars group analysis
        group_stats = pl_df.group_by(category_col).agg([
            pl.col(numeric_col).mean().alias('group_mean'),
            pl.col(numeric_col).count().alias('group_count')
        ])

        # SciPy statistical tests
        groups = [group[numeric_col].to_numpy() for name, group in pd_df.groupby(category_col)]
        if len(groups) >= 2:
            f_stat, anova_p = stats.f_oneway(*groups)
        else:
            f_stat, anova_p = None, None

        # Seaborn visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(data=pd_df, x=category_col, y=numeric_col, ax=axes[0])
        axes[0].set_title(f'{numeric_col} by {category_col}')

        sns.histplot(data=pd_df, x=numeric_col, hue=category_col, ax=axes[1])
        axes[1].set_title(f'{numeric_col} Distribution')

        plt.tight_layout()
        plt.show()

        # Return results
        return {
            'numpy_stats': np_stats,
            'group_analysis': group_stats,
            'anova': {'f_stat': f_stat, 'p_value': anova_p}
        }
    return (comprehensive_analysis,)


@app.cell
def _(comprehensive_analysis, np, pl):
    # Example usage of the comprehensive function
    sample_data_func = pl.DataFrame({
        'score': np.random.normal(75, 15, 200),
        'group': np.random.choice(['A', 'B', 'C'], 200)
    })

    results = comprehensive_analysis(sample_data_func, 'score', 'group')
    print("Comprehensive Analysis Results:")
    print(f"Overall statistics: {results['numpy_stats']}")
    print(f"ANOVA results: F={results['anova']['f_stat']:.3f}, p={results['anova']['p_value']:.3f}")

    print("\n" + "="*50)
    print("All code snippets completed successfully!")
    print("Each section above can be copied and pasted individually.")
    print("="*50)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook demonstrated:

    1. **Statistics Operations**: Descriptive statistics and visualization
    2. **Statistical Tests**: Hypothesis testing with t-tests
    3. **Tabular Operations**: Pandas vs Polars comparison
    4. **Data Types**: Memory usage and type systems

    Key insights:
    - Both pandas and polars are powerful for data analysis
    - Proper statistical testing requires understanding assumptions
    - Data type optimization can reduce memory usage
    - Visualization helps understand data patterns
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Python Statistics, Analytics & Visualization Notebook Summary

    ## Contents Overview

    ### Setup
    - Import numpy, polars, scipy, seaborn, matplotlib
    - Use `sns.set()` shortcut for quick seaborn styling
    - **Takeaway**: One-line setup for publication-ready visualizations

    ### NumPy basics
    - Basic statistics: mean, median, std, variance
    - Random data generation with seeds for reproducibility
    - Array operations: element-wise multiplication, dot products, correlation
    - **Takeaway**: NumPy is the foundation for numerical computing in Python - fast and memory-efficient for numerical arrays

    ### Polars vs pandas
    - Data type inference differences (Polars more efficient by default)
    - Memory usage comparison (Polars uses 20-30% less memory)
    - Performance benchmarks (Polars 2-5x faster for common operations)
    - Null handling (Polars maintains types, pandas promotes to float/object)
    - Type safety and schema enforcement
    - **Takeaway**: Polars is faster, uses less memory, and costs less in cloud environments - switch for large datasets and production pipelines

    ### Polars data manipulation
    - DataFrame creation and basic operations
    - Descriptive statistics with method chaining
    - Group by operations and aggregations
    - Filtering and transformations with expressions
    - **Takeaway**: Polars' expression API is more intuitive and consistent than pandas, with better performance

    ###  SciPy satistical tests and distributions
    - Independent samples t-test for comparing groups
    - Chi-square test for categorical associations
    - Shapiro-Wilk test for normality checking
    - Linear regression with correlation metrics
    - Distribution fitting (exponential, normal, etc.)
    - **Takeaway**: SciPy provides production-ready statistical tests - always check assumptions before applying tests

    ### Seaborn visualizations
    - Scatter plots with categories, sizes, and colors
    - Distribution plots with KDE overlays
    - Box plots for comparing groups
    - Correlation heatmaps with annotations
    - Pair plots for multivariate exploration
    - **Takeaway**: Seaborn makes complex visualizations simple - great for exploratory data analysis and presentations

    ### Advanced analytics combinations
    - Multi-library workflow for business analysis
    - Regional sales analysis with statistical tests
    - Subplot layouts for comprehensive reporting
    - **Takeaway**: Combine libraries strategically - each has strengths for different parts of the analysis pipeline

    ###  Hypothesis testing
    - Complete A/B test example with control and treatment groups
    - Descriptive statistics for both groups
    - T-test with p-values and effect sizes (Cohen's d)
    - Visualization with box plots and strip plots
    - **Takeaway**: Always report effect sizes along with p-values - statistical significance ≠ practical significance

    ### Custom functions
    - Reusable analysis function using all four libraries
    - Automated reporting with statistics and visualizations
    - ANOVA for multiple group comparisons
    - **Takeaway**: Build reusable functions for common analyses - saves time and ensures consistency

    ## Key takeaways

    ### NumPy

    Foundation for numerical computing - all other libraries build on it
    Extremely fast for vectorized operations
    Use for mathematical operations, random sampling, and array computations

    ### Polars

    Faster and cheaper than pandas - 2-5x performance gains, 20-30% less memory
    Better type system prevents silent errors
    Ideal for large datasets and production pipelines
    Lower cloud computing costs due to efficiency

    ### SciPy

    Gold standard for statistical tests and probability distributions
    Essential for hypothesis testing and scientific computing
    Always check test assumptions (normality, independence, equal variance)

    ### Seaborn

    Best for statistical visualizations with minimal code
    Built on matplotlib but much simpler API
    sns.set() instantly improves plot aesthetics
    Excellent for exploratory data analysis
    """
    )
    return


if __name__ == "__main__":
    app.run()
