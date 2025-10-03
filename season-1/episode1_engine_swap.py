#!/usr/bin/env python3
"""
Episode 1: The Data Mechanic's Garage – Engine Swap
===================================================

Transactions mini-pipeline comparing:
- Classic pandas
- pandas + PyArrow backend
- Polars (lazy + expressions)

Concepts demonstrated:
- Columnar engines & Arrow dtypes (dtypes, memory, parity)
- Vectorization vs apply/map (timings)
- Lazy evaluation & query planning (predicate/projection pushdown)
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
    # The Data Mechanic's Garage 🔧

    We're taking a classic transactions script into the garage and swapping engines:
    1) **Classic pandas** → our reliable baseline
    2) **pandas + PyArrow** → bolt-on upgrade (columnar dtypes)
    3) **Polars** → full engine swap (lazy, query planning, parallelism)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    import os
    import time
    import math
    import numpy as np
    import pandas as pd
    import polars as pl
    from pathlib import Path

    np.random.seed(42)
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "transactions_synthetic.csv"
    return Path, csv_path, np, os, pd, pl, time


@app.cell
def _(csv_path, np, os, pd):
    # Generate synthetic transactions once (idempotent)
    if not os.path.exists(csv_path):
        n_rows = 200_000
        days = np.random.randint(0, 365, size=n_rows)
        dates = np.datetime64("2024-01-01") + days.astype("timedelta64[D]")
        regions = np.random.choice(["North", "South", "East", "West"], size=n_rows)
        promo_pool = np.array([None, "PROMO10", "PROMO20", "PROMO30", None, None], dtype=object)

        df_gen = pd.DataFrame({
            "date": dates.astype("datetime64[ns]"),
            "customer_id": np.random.randint(1, 50_001, size=n_rows),
            "product_id": np.random.randint(1, 501, size=n_rows),
            "price": np.round(np.random.gamma(shape=2.0, scale=15.0, size=n_rows), 2),
            "quantity": np.random.randint(1, 6, size=n_rows),
            "region": regions,
            "promo_code": np.random.choice(promo_pool, size=n_rows)
        })

        df_gen.to_csv(csv_path, index=False)
        print(f"✅ Generated synthetic CSV: {csv_path}")
    else:
        print(f"ℹ️ Using existing CSV: {csv_path}")
    return


@app.cell
def _(np, pd):
    # Small product lookup
    product_ids = np.arange(1, 501)
    categories = np.array(["Electronics", "Home", "Clothing", "Sports", "Grocery"])
    product_cat = pd.DataFrame({
        "product_id": product_ids,
        "category": categories[(product_ids - 1) % len(categories)]
    })
    return (product_cat,)


@app.cell
def _(mo):
    mo.md(r"""## Columnar engines and Arrow dtypes""")
    return


@app.cell
def _(csv_path, pd, pl):
    # Classic pandas read
    df_pd = pd.read_csv(csv_path, parse_dates=["date"])  # NumPy dtypes
    mem_pd_mb = df_pd.memory_usage(deep=True).sum() / 1024**2
    checksum_pd = (df_pd["price"] * df_pd["quantity"]).sum()

    # pandas + PyArrow backend
    df_pa = pd.read_csv(
        csv_path,
        engine="pyarrow",
        dtype_backend="pyarrow",
        parse_dates=["date"],
    )
    mem_pa_mb = df_pa.memory_usage(deep=True).sum() / 1024**2
    checksum_pa = (df_pa["price"] * df_pa["quantity"]).sum()

    # Polars eager read
    df_pl = pl.read_csv(csv_path)
    if df_pl.get_column("date").dtype != pl.Date and df_pl.get_column("date").dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    mem_pl_mb = df_pl.estimated_size() / 1024**2
    checksum_pl = (df_pl.get_column("price") * df_pl.get_column("quantity")).sum()

    print("=== DTYPE SNAPSHOT ===")
    print("pandas (classic):")
    print(df_pd.dtypes)
    print("\npandas (PyArrow backend):")
    print(df_pa.dtypes)
    print("\npolars:")
    print(df_pl.dtypes)

    print("\n=== MEMORY (MB) ===")
    print(f"pandas classic: {mem_pd_mb:.2f}")
    print(f"pandas + PyArrow: {mem_pa_mb:.2f}")
    print(f"polars: {mem_pl_mb:.2f}")

    print("\n=== CHECKSUM PARITY (revenue sum) ===")
    print(f"pandas classic: {checksum_pd:.2f}")
    print(f"pandas + PyArrow: {checksum_pa:.2f}")
    print(f"polars: {checksum_pl:.2f}")

    parity_checksum_ok = (
        abs(checksum_pd - checksum_pa) < 1e-6 and
        abs(float(checksum_pd) - float(checksum_pl)) < 1e-6
    )
    print(f"Parity OK: {parity_checksum_ok}")
    return (df_pd,)


@app.cell
def _(mo):
    mo.md(r"""## Vectorization vs apply/map""")
    return


@app.cell
def _(df_pd, pl, time):
    def _run_vectorization_bench():
        _df_small = df_pd.copy()

        # Row-wise apply
        _t0 = time.perf_counter()
        _ = _df_small.apply(lambda r: r["price"] * r["quantity"], axis=1)
        _dur_apply = time.perf_counter() - _t0

        # Vectorized pandas
        _t0 = time.perf_counter()
        _ = _df_small["price"] * _df_small["quantity"]
        _dur_vec = time.perf_counter() - _t0

        # Polars expression
        _lf_expr = pl.from_pandas(_df_small).lazy()
        _t0 = time.perf_counter()
        _ = _lf_expr.with_columns((pl.col("price") * pl.col("quantity")).alias("revenue")).collect()
        _dur_pl_expr = time.perf_counter() - _t0

        print("Row-wise apply (pandas): {:.4f}s".format(_dur_apply))
        print("Vectorized (pandas):     {:.4f}s".format(_dur_vec))
        print("Expression (polars):     {:.4f}s".format(_dur_pl_expr))

    _run_vectorization_bench()
    return


@app.cell
def _(mo):
    mo.md(r"""## Lazy evaluation and query planning""")
    return


@app.cell
def _(csv_path, pl):
    # Toggle to showcase pushdown effects
    filter_on = True

    # Build lazy pipeline
    _lf_plan = (
        pl.scan_csv(csv_path)
        .with_columns([
            (pl.col("price") * pl.col("quantity")).alias("revenue"),
            pl.col("promo_code").fill_null("").str.starts_with("PROMO").alias("has_promo"),
            pl.col("date").str.strptime(pl.Date, strict=False).dt.truncate("1mo").alias("month"),
        ])
        .select(["region", "month", "revenue", "has_promo"])
    )

    if filter_on:
        _lf_plan = _lf_plan.filter(pl.col("region") == "North")

    print("=== Polars Query Plan ===")
    print(_lf_plan.explain())

    out = _lf_plan.collect()
    print(f"Rows after filter: {out.height}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Mini-pipeline: group, join, top categories""")
    return


@app.cell
def _(csv_path, pd, pl, product_cat, time):
    # Helper: pandas pipeline
    def run_pandas(use_pyarrow: bool) -> tuple[float, "pd.DataFrame"]:
        read_kwargs = dict(parse_dates=["date"]) if not use_pyarrow else dict(parse_dates=["date"], engine="pyarrow", dtype_backend="pyarrow")
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path, **read_kwargs)
        # Robust month extraction for both numpy- and Arrow-backed datetimes
        try:
            df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        except Exception:
            # Arrow-backed datetime lacks to_period; fallback via year-month normalization
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        df["revenue"] = df["price"] * df["quantity"]
        df["has_promo"] = df["promo_code"].fillna("").str.startswith("PROMO")
        df = df.merge(product_cat, on="product_id", how="left")

        grouped = (
            df.groupby(["region", "month", "category"], as_index=False)
              .agg(total_revenue=("revenue", "sum"),
                   avg_price=("price", "mean"),
                   distinct_customers=("customer_id", "nunique"))
        )

        topk = (
            grouped.sort_values(["region", "total_revenue"], ascending=[True, False])
                   .groupby("region", as_index=False)
                   .head(5)
        )
        dur = time.perf_counter() - t0
        return dur, topk

    # Helper: polars pipeline
    def run_polars() -> tuple[float, "pl.DataFrame"]:
        t0 = time.perf_counter()
        _lf_full = (
            pl.scan_csv(csv_path)
            .with_columns([
                (pl.col("price") * pl.col("quantity")).alias("revenue"),
                pl.col("date").str.strptime(pl.Date, strict=False).dt.truncate("1mo").alias("month"),
                pl.col("promo_code").fill_null("").str.starts_with("PROMO").alias("has_promo"),
            ])
            .join(pl.from_pandas(product_cat).lazy(), on="product_id", how="left")
            .group_by(["region", "month", "category"]).agg([
                pl.col("revenue").sum().alias("total_revenue"),
                pl.col("price").mean().alias("avg_price"),
                pl.col("customer_id").n_unique().alias("distinct_customers"),
            ])
            .sort(["region", "total_revenue"], descending=[False, True])
            .group_by("region")
            .head(5)
        )
        out = _lf_full.collect()
        dur = time.perf_counter() - t0
        return dur, out

    # Run all
    dur_pd, res_pd = run_pandas(use_pyarrow=False)
    dur_pa, res_pa = run_pandas(use_pyarrow=True)
    dur_pl, res_pl = run_polars()

    # Parity checks (coarse)
    tot_pd = res_pd["total_revenue"].sum()
    tot_pa = res_pa["total_revenue"].sum()
    tot_pl = res_pl.get_column("total_revenue").sum()
    parity_pipeline_ok = abs(tot_pd - tot_pa) < 1e-6 and abs(float(tot_pd) - float(tot_pl)) < 1e-6

    # Scoreboard
    scoreboard = pl.DataFrame({
        "engine": ["pandas", "pandas+pyarrow", "polars"],
        "time_sec": [dur_pd, dur_pa, dur_pl],
        "rows_out": [len(res_pd), len(res_pa), res_pl.height],
    }).with_columns(pl.col("time_sec").round(4))

    print("=== Mini-pipeline scoreboard ===")
    print(scoreboard)
    print(f"Parity OK (total_revenue): {parity_pipeline_ok}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Takeaways

    - **Columnar engines** reduce memory and improve IO; Arrow dtypes modernize pandas' internals.
    - **Vectorization** beats row-wise apply; Polars expressions keep logic fast and declarative.
    - **Lazy evaluation** lets Polars optimize the whole plan and push filters down to IO.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Large-scale benchmark with NYC TLC Parquet""")
    return


@app.cell
def _(Path):
    # Download a few NYC TLC Parquet files (public HTTP); no AWS creds needed
    import urllib.request as _urlreq

    _tlc_base = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    _tlc_dir = Path(__file__).parent / "data" / "tlc"
    _tlc_dir.mkdir(parents=True, exist_ok=True)

    # Choose a small set to start; scale by adding more months
    _tlc_names = [
        "yellow_tripdata_2024-01.parquet",
        "yellow_tripdata_2024-02.parquet",
    ]

    tlc_files = []
    for _name in _tlc_names:
        _url = f"{_tlc_base}/{_name}"
        _local = _tlc_dir / _name
        if not _local.exists():
            _urlreq.urlretrieve(_url, _local)
            print(f"⬇️  {_local.name}")
        else:
            print(f"ℹ️ Exists: {_local.name}")
        tlc_files.append(str(_local))
    return (tlc_files,)


@app.cell
def _(pd, pl, time, tlc_files):
    # Benchmark: read Parquet and count rows; compare pandas vs polars
    def _safe_sum_pd(df, cols):
        for c in cols:
            if c in df.columns:
                return float(df[c].sum())
        return float("nan")

    def _safe_sum_pl(df, cols):
        for c in cols:
            if c in df.columns:
                return float(df.select(pl.col(c).sum()).item())
        return float("nan")

    def _run_tlc_bench():
        _rows_pd_np, _rows_pd_pa, _rows_pl_cpu = [], [], []
        _t_pd_np, _t_pd_pa, _t_pl_cpu = [], [], []
        _sum_pd_np, _sum_pd_pa, _sum_pl_cpu = [], [], []

        for _f in tlc_files:
            # pandas (default engine)
            _t0 = time.perf_counter()
            _dfpd_np = pd.read_parquet(_f)
            _t_pd_np.append(time.perf_counter() - _t0)
            _rows_pd_np.append(len(_dfpd_np))
            _sum_pd_np.append(_safe_sum_pd(_dfpd_np, ["fare_amount", "total_amount"]))

            # pandas + pyarrow
            _t0 = time.perf_counter()
            _dfpd_pa = pd.read_parquet(_f, engine="pyarrow")
            _t_pd_pa.append(time.perf_counter() - _t0)
            _rows_pd_pa.append(len(_dfpd_pa))
            _sum_pd_pa.append(_safe_sum_pd(_dfpd_pa, ["fare_amount", "total_amount"]))

            # polars CPU (eager)
            _t0 = time.perf_counter()
            _dfpl_cpu = pl.read_parquet(_f)
            _t_pl_cpu.append(time.perf_counter() - _t0)
            _rows_pl_cpu.append(_dfpl_cpu.height)
            _sum_pl_cpu.append(_safe_sum_pl(_dfpl_cpu, ["fare_amount", "total_amount"]))


        # Set Pandas option to display all columns
        _bench = pl.DataFrame({
            "file": tlc_files,
            "rows_pd_np": _rows_pd_np,
            "time_pd_np_s": _t_pd_np,
            "rows_pd_pa": _rows_pd_pa,
            "time_pd_pa_s": _t_pd_pa,
            "rows_pl_cpu": _rows_pl_cpu,
            "time_pl_cpu_s": _t_pl_cpu,
            "sum_fare_pd_np": _sum_pd_np,
            "sum_fare_pd_pa": _sum_pd_pa,
            "sum_fare_pl_cpu": _sum_pl_cpu,
        })

        print("=== NYC TLC Parquet read benchmark (pandas default vs pandas+pyarrow vs polars CPU) ===")
        with pl.Config(tbl_cols=10):
            print(_bench)

    _run_tlc_bench()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## High‑signal aggregation benchmark (where Polars shines)""")
    return


@app.cell
def _(pd, pl, time, tlc_files):
    # Scenario: projection + filter + groupby over multiple files
    # Columns used are common in Yellow Taxi schema
    _cols = ["PULocationID", "trip_distance", "fare_amount", "passenger_count"]

    def _pandas_groupby(engine: str | None) -> tuple[float, "pd.DataFrame"]:
        _groups = []
        _t0 = time.perf_counter()
        for _f in tlc_files:
            _df = pd.read_parquet(_f, columns=_cols, engine=engine) if engine else pd.read_parquet(_f, columns=_cols)
            _df = _df[_df["trip_distance"] > 2]
            _df = _df[_df["passenger_count"] >= 1]
            _g = (
                _df.groupby("PULocationID", as_index=False)
                   .agg(total_fare=("fare_amount", "sum"),
                        avg_distance=("trip_distance", "mean"),
                        rides=("fare_amount", "size"))
            )
            _groups.append(_g)
        _out = pd.concat(_groups, ignore_index=True)
        _out = _out.groupby("PULocationID", as_index=False).agg({
            "total_fare": "sum",
            "avg_distance": "mean",
            "rides": "sum",
        }).sort_values("total_fare", ascending=False).head(10)
        _dur = time.perf_counter() - _t0
        return _dur, _out

    def _polars_groupby(engine: str | None) -> tuple[float, "pl.DataFrame"]:
        _lazy = (
            pl.scan_parquet(tlc_files)
            .select(_cols)
            .filter((pl.col("trip_distance") > 2) & (pl.col("passenger_count") >= 1))
            .group_by("PULocationID")
            .agg([
                pl.col("fare_amount").sum().alias("total_fare"),
                pl.col("trip_distance").mean().alias("avg_distance"),
                pl.count().alias("rides"),
            ])
            .sort("total_fare", descending=True)
            .head(10)
        )
        _t0 = time.perf_counter()
        _out = _lazy.collect(engine=engine) if engine else _lazy.collect()
        _dur = time.perf_counter() - _t0
        return _dur, _out

    # Run three variants
    _t_pd_np, _pd_np = _pandas_groupby(engine=None)
    _t_pd_pa, _pd_pa = _pandas_groupby(engine="pyarrow")
    _t_pl_cpu, _pl_cpu = _polars_groupby(engine=None)

    # Optional GPU (best-effort; may fall back silently depending on environment)
    try:
        _t_pl_gpu, _pl_gpu = _polars_groupby(engine="gpu")
        _gpu_row = {"engine": "polars_lazy_gpu", "time_s": _t_pl_gpu}
    except Exception:
        _pl_gpu = None
        _gpu_row = {"engine": "polars_lazy_gpu", "time_s": float("nan")}

    _score = pl.DataFrame([
        {"engine": "pandas_default", "time_s": _t_pd_np},
        {"engine": "pandas_pyarrow", "time_s": _t_pd_pa},
        {"engine": "polars_lazy_cpu", "time_s": _t_pl_cpu},
        _gpu_row,
    ]).with_columns(pl.col("time_s").round(3))

    print("=== Aggregation benchmark: projection + filter + groupby (top 10 by total_fare) ===")
    print(_score)
    print("\nPolars result head (CPU):")
    print(_pl_cpu)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
