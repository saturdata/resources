#!/usr/bin/env python3
"""
Data Visualization in Python - Marimo Notebook
==============================================
Covers Seaborn plotting, matplotlib styling, and visualization best practices
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
    # Data Visualization in Python
    Saturdata: Season 1

    This notebook focuses on creating effective visualizations including:
    - Seaborn statistical plotting functions
    - Plotly interactive charts and dashboards
    - Matplotlib styling and customization  
    - Best practices for business presentations
    - Converting between Polars and pandas for visualization
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    # Import required libraries for data visualization
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import warnings

    warnings.filterwarnings("ignore")

    print("✅ Data visualization libraries imported successfully")
    return go, np, pl, plt, px, sns


@app.cell
def _(sns):
    # Set seaborn style for all charts with one quick shortcut
    sns.set()
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
    viz_data = pl.DataFrame(
        {
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "value": np.random.exponential(2, 100),
        }
    )
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
    sns.scatterplot(data=viz_data_pd, x="x", y="y", hue="category", size="value")
    plt.title("Scatter Plot with Categories and Sizes")
    plt.show()
    return


@app.cell
def _(plt, sns, viz_data_pd):
    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=viz_data_pd, x="value", hue="category", kde=True)
    plt.title("Distribution by Category")
    plt.show()
    return


@app.cell
def _(plt, sns, viz_data_pd):
    # Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=viz_data_pd, x="category", y="value")
    plt.title("Box Plot by Category")
    plt.show()
    return


@app.cell
def _(np, plt, sns, viz_data_pd):
    # Correlation heatmap
    correlation_data = viz_data_pd.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()
    return


@app.cell
def _(np, pl, plt, sns):
    # Pair plot for multiple variables
    sample_df = pl.DataFrame(
        {
            "height": np.random.normal(170, 10, 100),
            "weight": np.random.normal(70, 15, 100),
            "age": np.random.randint(18, 65, 100),
            "gender": np.random.choice(["M", "F"], 100),
        }
    ).to_pandas()

    sns.pairplot(sample_df, hue="gender")
    plt.suptitle("Pair Plot by Gender", y=1.02)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The power of sns.set()
    How seaborn styling affects all matplotlib plots
    """
    )
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
    x_data = viz_data_pd["x"].values
    y_data = viz_data_pd["y"].values
    axes_mpl[0].plot(x_data, y_data, "o-", linewidth=2, markersize=6)
    axes_mpl[0].set_xlabel("X Values")
    axes_mpl[0].set_ylabel("Y Values")
    axes_mpl[0].set_title("Matplotlib Plot with Seaborn Theme")
    axes_mpl[0].grid(True, alpha=0.3)

    # Pure matplotlib bar chart - also styled by seaborn
    category_counts = viz_data_pd["category"].value_counts()
    axes_mpl[1].bar(category_counts.index, category_counts.values, alpha=0.8)
    axes_mpl[1].set_xlabel("Category")
    axes_mpl[1].set_ylabel("Count")
    axes_mpl[1].set_title("Matplotlib Bar Chart with Seaborn Theme")

    plt.tight_layout()
    plt.show()

    print("Notice: Regular matplotlib plots now have seaborn's clean aesthetic!")
    print("Benefits: Better colors, nicer grid, improved fonts, professional look")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Business visualization example
    Multi-panel layouts for comprehensive analysis
    """
    )
    return


@app.cell
def _(np, pl):
    # Generate business dataset for visualization
    np.random.seed(42)
    business_data = pl.DataFrame(
        {
            "sales": np.random.gamma(2, 1000, 500),
            "marketing_spend": np.random.uniform(100, 1000, 500),
            "region": np.random.choice(["North", "South", "East", "West"], 500),
            "quarter": np.random.choice(["Q1", "Q2", "Q3", "Q4"], 500),
        }
    )
    return (business_data,)


@app.cell
def _(business_data, plt, sns):
    # Visualization of the business analysis
    business_pd = business_data.to_pandas()

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Sales distribution by region
    sns.boxplot(data=business_pd, x="region", y="sales", ax=axes[0, 0])
    axes[0, 0].set_title("Sales by Region")

    # Marketing spend vs sales correlation
    sns.scatterplot(
        data=business_pd, x="marketing_spend", y="sales", hue="region", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Marketing Spend vs Sales")

    # Quarterly trends
    sns.barplot(data=business_pd, x="quarter", y="sales", ax=axes[1, 0])
    axes[1, 0].set_title("Sales by Quarter")

    # Distribution of sales
    sns.histplot(data=business_pd, x="sales", kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Sales Distribution")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## A/B test visualization
    Effective ways to show statistical comparisons
    """
    )
    return


@app.cell
def _(control_group, np, pl, plt, sns, treatment_group):
    # Visualization
    ab_data = pl.DataFrame(
        {
            "group": ["Control"] * 100 + ["Treatment"] * 100,
            "value": np.concatenate([control_group, treatment_group]),
        }
    ).to_pandas()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ab_data, x="group", y="value")
    sns.stripplot(data=ab_data, x="group", y="value", alpha=0.5, size=3)
    plt.title("A/B Test Results")
    plt.show()
    return


@app.cell
def _(np):
    # Generate A/B test data for visualization
    np.random.seed(42)
    control_group = np.random.normal(2.3, 0.5, 100)
    treatment_group = np.random.normal(2.8, 0.6, 100)
    return control_group, treatment_group


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Plotly: Interactive visualizations
    Modern, interactive charts for web-based analytics
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Plotly creates interactive, web-based visualizations that allow users to hover, zoom, pan, and explore data dynamically. Unlike static matplotlib/seaborn charts, Plotly charts are JavaScript-based and perfect for dashboards, reports, and exploratory analysis where interactivity enhances understanding.""")
    return


@app.cell
def _(pl):
    # Load the transactions data for Plotly examples
    transactions = pl.read_csv("data/transactions_synthetic.csv")

    # Parse date and add derived columns for better visualizations
    transactions_enriched = transactions.with_columns(
        [
            pl.col("date").str.to_date().alias("date"),
            (pl.col("price") * pl.col("quantity")).alias("revenue"),
        ]
    )

    transactions_enriched.head()
    return (transactions_enriched,)


@app.cell
def _(px, transactions_enriched):
    # Interactive scatter plot: Price vs Quantity by Region
    # Using first 100 values because the full dataset would be too large
    fig_scatter = px.scatter(
        transactions_enriched[0:100].to_pandas(),
        x="price",
        y="quantity",
        color="region",
        size="revenue",
        hover_data=["product_id", "promo_code"],
        title="Transaction Analysis: Price vs Quantity by Region",
        labels={"price": "Price ($)", "quantity": "Quantity"},
        opacity=0.6,
    )

    fig_scatter.update_layout(height=500, template="plotly_white")

    fig_scatter
    return


@app.cell
def _(pl, px, transactions_enriched):
    # Interactive bar chart: Revenue by Region
    revenue_by_region = (
        transactions_enriched.group_by("region")
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .sort("total_revenue", descending=True)
    )

    fig_bar = px.bar(
        revenue_by_region.to_pandas(),
        x="region",
        y="total_revenue",
        title="Total Revenue by Region",
        labels={"total_revenue": "Total Revenue ($)", "region": "Region"},
        color="total_revenue",
        color_continuous_scale="blues",
        text="total_revenue",
    )

    fig_bar.update_traces(texttemplate="$%{text:.2s}", textposition="outside")
    fig_bar.update_layout(height=500, showlegend=False, template="plotly_white")

    fig_bar
    return


@app.cell
def _(pl, px, transactions_enriched):
    # Time series: Daily revenue trends
    daily_revenue = (
        transactions_enriched.group_by("date")
        .agg(pl.col("revenue").sum().alias("daily_revenue"))
        .sort("date")
    )

    fig_timeseries = px.line(
        daily_revenue.to_pandas(),
        x="date",
        y="daily_revenue",
        title="Daily Revenue Trends",
        labels={"daily_revenue": "Daily Revenue ($)", "date": "Date"},
        markers=True,
    )

    fig_timeseries.update_layout(
        height=500, template="plotly_white", hovermode="x unified"
    )

    fig_timeseries
    return


@app.cell
def _(px, transactions_enriched):
    # Box plot: Price distribution by region
    fig_box = px.box(
        transactions_enriched.to_pandas(),
        x="region",
        y="price",
        color="region",
        title="Price Distribution by Region",
        labels={"price": "Price ($)", "region": "Region"},
        points="outliers",  # Show outlier points
    )

    fig_box.update_layout(height=500, template="plotly_white", showlegend=False)

    fig_box
    return


@app.cell
def _(go, pl, transactions_enriched):
    # Advanced: Sunburst chart showing revenue hierarchy by Region > Promo Code
    revenue_hierarchy = (
        transactions_enriched.filter(pl.col("promo_code") != "")
        .group_by(["region", "promo_code"])
        .agg(pl.col("revenue").sum().alias("revenue"))
    ).to_pandas()

    # Calculate region totals for parent nodes
    region_totals = revenue_hierarchy.groupby("region")["revenue"].sum().reset_index()

    # Build the sunburst data structure
    # Parents: regions (with empty string parent to be root nodes)
    # Children: promo codes (with region as parent)
    labels = list(region_totals["region"]) + list(revenue_hierarchy["promo_code"])
    parents = [""] * len(region_totals) + list(revenue_hierarchy["region"])
    values = list(region_totals["revenue"]) + list(revenue_hierarchy["revenue"])

    fig_sunburst = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colorscale="Blues"),
        )
    )

    fig_sunburst.update_layout(
        title="Revenue Breakdown: Region and Promo Code",
        height=600,
        template="plotly_white",
    )

    fig_sunburst
    return


@app.cell
def _(pl, px, transactions_enriched):
    # Heatmap: Average revenue by region and month
    transactions_with_month = transactions_enriched.with_columns(
        pl.col("date").dt.month().alias("month")
    )

    revenue_heatmap = (
        transactions_with_month.group_by(["region", "month"])
        .agg(pl.col("revenue").mean().alias("avg_revenue"))
        .sort(["region", "month"])
    )

    # Pivot for heatmap
    heatmap_pivot = revenue_heatmap.to_pandas().pivot(
        index="region", columns="month", values="avg_revenue"
    )

    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month", y="Region", color="Avg Revenue ($)"),
        title="Average Revenue by Region and Month",
        aspect="auto",
        color_continuous_scale="YlOrRd",
    )

    fig_heatmap.update_layout(height=500)

    fig_heatmap
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Key Plotly Benefits:**

    - **Interactivity**: Hover to see data points, zoom into regions, pan across time
    - **Modern Look**: Professional, web-ready visualizations without manual styling
    - **Rich Chart Types**: From basic plots to advanced 3D and geographic visualizations
    - **Easy Export**: Save as interactive HTML or static images
    - **Marimo Integration**: Charts render natively in marimo notebooks
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Advanced visualization techniques
    Professional presentation tips
    """
    )
    return


@app.cell
def _(np, pl, plt, sns):
    # Advanced styling and customization
    advanced_data = pl.DataFrame(
        {
            "metric": np.random.normal(100, 15, 200),
            "segment": np.random.choice(["Premium", "Standard", "Budget"], 200),
            "time_period": np.random.choice(["Q1", "Q2", "Q3", "Q4"], 200),
        }
    ).to_pandas()

    # Create a professional-looking visualization
    plt.figure(figsize=(12, 8))

    # Use a professional color palette
    colors = sns.color_palette("Set2", n_colors=3)

    # Create violin plots with custom styling
    sns.violinplot(
        data=advanced_data, x="segment", y="metric", palette=colors, inner="box"
    )

    # Add titles and labels with professional formatting
    plt.title(
        "Performance Metrics by Customer Segment",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Customer Segment", fontsize=12, fontweight="bold")
    plt.ylabel("Performance Metric", fontsize=12, fontweight="bold")

    # Improve the overall appearance
    plt.grid(True, alpha=0.3, axis="y")
    sns.despine()  # Remove top and right spines

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook demonstrated:

    1. **Seaborn Plotting**: Statistical visualizations with minimal code
    2. **Styling Control**: How `sns.set()` transforms all matplotlib plots
    3. **Multi-panel Layouts**: Comprehensive business analysis dashboards
    4. **Plotly Interactive Charts**: Modern, web-based visualizations with hover, zoom, and pan
    5. **Professional Presentation**: Color palettes, formatting, and best practices
    6. **Data Conversion**: Efficient Polars-to-pandas workflows for visualization

    Key insights:
    - Seaborn dramatically reduces code required for complex visualizations
    - `sns.set()` instantly improves the appearance of all plots
    - Plotly enables interactive exploration and modern dashboards
    - Choose plot types that match your data story and audience needs
    - Convert from Polars to pandas only when needed for visualization
    - Professional styling makes data more persuasive and trustworthy
    """
    )
    return


if __name__ == "__main__":
    app.run()
