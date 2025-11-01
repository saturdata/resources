#!/usr/bin/env python3
"""
Statistical Testing in Python - Marimo Notebook
==============================================
Covers SciPy statistical tests, hypothesis testing, and distribution analysis
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
    # Statistical Testing in Python
    Saturdata: Season 1

    This notebook focuses on statistical analysis and hypothesis testing including:
    - SciPy statistical tests and distributions
    - Hypothesis testing with A/B test examples
    - Distribution fitting and normality testing
    - Correlation and regression analysis
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    # Import required libraries for statistical testing
    import numpy as np
    import pandas as pd
    import polars as pl
    import scipy.stats as stats
    from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency, pearsonr
    import warnings

    warnings.filterwarnings("ignore")

    print("✅ Statistical testing libraries imported successfully")
    return np, pd, pl, stats, ttest_1samp, ttest_ind, chi2_contingency, pearsonr


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
def _(np, pl):
    # Regional sales analysis with statistical tests
    np.random.seed(42)
    business_data = pl.DataFrame({
        'sales': np.random.gamma(2, 1000, 500),
        'marketing_spend': np.random.uniform(100, 1000, 500),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
        'quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 500)
    })
    
    north_sales = business_data.filter(pl.col('region') == 'North')['sales'].to_numpy()
    south_sales = business_data.filter(pl.col('region') == 'South')['sales'].to_numpy()

    # Perform statistical test for significant different between north and south sales
    t_stat_ns, p_val_ns = stats.ttest_ind(north_sales, south_sales)
    print(f"North vs South sales t-test: t={t_stat_ns:.3f}, p={p_val_ns:.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ANOVA for multiple groups
    Testing differences between more than two groups
    """
    )
    return


@app.cell
def _(business_data, pd, stats):
    # ANOVA for multiple group comparisons
    business_pd = business_data.to_pandas()
    groups = [group['sales'].to_numpy() for name, group in business_pd.groupby('region')]
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"ANOVA test across all regions: F={f_stat:.3f}, p={anova_p:.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook demonstrated:

    1. **Statistical Tests**: t-tests, chi-square, ANOVA for comparing groups
    2. **Normality Testing**: Shapiro-Wilk test and practical considerations
    3. **Hypothesis Testing**: A/B test analysis with effect sizes
    4. **Distribution Fitting**: Parameter estimation for theoretical distributions
    5. **Regression Analysis**: Linear relationships and correlation

    Key insights:
    - Always report effect sizes alongside p-values
    - Statistical significance ≠ practical significance  
    - Real-world data rarely passes normality tests, but we proceed anyway
    - Choose the right test based on data type and research question
    """
    )
    return


if __name__ == "__main__":
    app.run()