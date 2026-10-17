"""Generate a synthetic dataset in the exact Online Retail II format.

Used ONLY for testing the pipeline when the real Kaggle file is not available.
The real dataset should always be used for your final results.
"""
import numpy as np
import pandas as pd


def make_sample(n_customers=700, n_products=180, start="2010-12-01", end="2011-12-09", seed=42):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    products = pd.DataFrame({
        "StockCode": [f"{20000 + i}" for i in range(n_products)],
        "Description": [f"GIFT ITEM {i:03d}" for i in range(n_products)],
        "Price": np.round(rng.lognormal(1.0, 0.6, n_products).clip(0.29, 40), 2),
        "pop": rng.dirichlet(np.ones(n_products) * 0.4),
    })
    cust_ids = np.arange(12346, 12346 + n_customers)
    # customers differ in activity level and lifetime (some stop buying -> churners)
    activity = rng.gamma(1.5, 1.0, n_customers)
    first = rng.integers(0, 200, n_customers)
    last = np.where(rng.random(n_customers) < 0.3, rng.integers(first + 30, len(days), n_customers), len(days))
    countries = np.where(rng.random(n_customers) < 0.88, "United Kingdom",
                         rng.choice(["Germany", "France", "EIRE", "Spain", "Netherlands"], n_customers))

    rows, inv = [], 580000
    t = np.arange(len(days))
    trend = 1 + 1.2 * (t / len(days)) ** 2          # grows into the Q4 peak
    weekly = np.array([0.0, 1.0, 1.1, 1.0, 1.15, 0.8, 0.0, ])  # Mon..Sun, no Saturday trading like the real data
    for d_i, d in enumerate(days):
        w = weekly[d.dayofweek]
        if w == 0:
            continue
        season = 1 + 0.5 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365)
        lam = 0.012 * w * trend[d_i] * season
        active = (first <= d_i) & (d_i < last)
        p = np.clip(lam * activity * active, 0, 0.9)
        buyers = np.where(rng.random(n_customers) < p)[0]
        for c in buyers:
            inv += 1
            n_lines = rng.integers(1, 12)
            ts = d + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
            items = rng.choice(n_products, n_lines, p=products["pop"] / products["pop"].sum())
            for it in items:
                qty = int(rng.choice([1, 2, 3, 4, 6, 12, 24], p=[.25, .2, .12, .1, .15, .12, .06]))
                rows.append((str(inv), products.StockCode[it], products.Description[it], qty, ts,
                             products.Price[it], int(cust_ids[c]), countries[c]))
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])

    # --- inject typical real-world mess ---
    n = len(df)
    # cancellations
    canc = df.sample(frac=0.02, random_state=seed).copy()
    canc["Invoice"] = "C" + canc["Invoice"]
    canc["Quantity"] = -canc["Quantity"]
    df = pd.concat([df, canc], ignore_index=True)
    # guest checkouts with no customer id
    miss = df.sample(frac=0.2, random_state=seed + 1).index
    df["Customer ID"] = df["Customer ID"].astype("float")
    df.loc[miss, "Customer ID"] = np.nan
    # bad prices / non-product codes / duplicates
    df.loc[df.sample(n=40, random_state=3).index, "Price"] = 0.0
    df.loc[df.sample(n=15, random_state=4).index, ["StockCode", "Description"]] = ["POST", "POSTAGE"]
    df.loc[df.sample(n=25, random_state=5).index, "Description"] = np.nan
    df = pd.concat([df, df.sample(n=300, random_state=6)], ignore_index=True)
    return df.sort_values("InvoiceDate").reset_index(drop=True)


if __name__ == "__main__":
    out = make_sample()
    print(out.shape)
    print(out.head())
