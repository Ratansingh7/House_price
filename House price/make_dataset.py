"""
Generate a realistic synthetic house-price dataset.

The generator emulates the structure of common real-estate benchmark datasets
(Ames / King County style): a multiplicative price model driven by size,
configuration, quality, location and time depreciation, with log-normal noise.

Output: data/houses.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 20240617
N_ROWS = 12_000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_PATH = os.path.join(OUT_DIR, "houses.csv")

NEIGHBORHOODS = {
    "Northwood": 1.28,
    "Riverside": 1.16,
    "Oakridge": 1.09,
    "Downtown": 1.34,
    "Hillcrest": 1.22,
    "Lakeview": 1.13,
    "Maplewood": 0.94,
    "Sunnyvale": 0.88,
    "Brookfield": 0.97,
    "Cedar Park": 0.83,
}

MAIN_ROAD = {"Y": 1.09, "N": 1.0}
GARAGE_SPOTS = {0: 0.97, 1: 1.0, 2: 1.06, 3: 1.11, 4: 1.14}
FLOOR_MARGINAL = {1: 1.00, 2: 1.07, 3: 1.13, 4: 1.18}
RENOVATION = {"Not_Renovated": 1.0, "Partial": 1.06, "Full": 1.14}
BASEMENT = {0: 1.0, 1: 1.11}
FURNISHED = {"N": 1.0, "Y": 1.06}


def _rounded_normal(rng, mean, std, lo, hi, size):
    return np.clip(rng.normal(mean, std, size), lo, hi).round()


def build_dataframe(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area = _rounded_normal(rng, 1850, 750, 480, 6200, n_rows)
    bedrooms = np.clip(rng.poisson(3.1, n_rows) + 1, 1, 8)
    bathrooms = np.clip(np.round(rng.poisson(2.0, n_rows) / 2 + 0.5, 1) + 0.5, 1, 6)
    floors = rng.choice([1, 2, 3, 4], n_rows, p=[0.12, 0.51, 0.28, 0.09])
    garage = rng.choice([0, 1, 2, 3, 4], n_rows, p=[0.18, 0.34, 0.31, 0.13, 0.04])
    has_basement = rng.choice([0, 1], n_rows, p=[0.86, 0.14])
    has_pool = rng.choice([0, 1], n_rows, p=[0.93, 0.07])
    air_conditioning = rng.choice([0, 1], n_rows, p=[0.55, 0.45])
    furnished = rng.choice([0, 1], n_rows, p=[0.88, 0.12])
    main_road = rng.choice([0, 1], n_rows, p=[0.42, 0.58])
    renovation = rng.choice(
        list(RENOVATION.keys()), n_rows, p=[0.68, 0.22, 0.10]
    )
    neighborhood = rng.choice(
        list(NEIGHBORHOODS.keys()),
        n_rows,
        p=[0.10, 0.11, 0.11, 0.09, 0.10, 0.11, 0.10, 0.10, 0.09, 0.09],
    )
    overall_quality = np.clip(_rounded_normal(rng, 5.4, 1.2, 1, 10, n_rows), 1, 10)
    year_built = np.clip(
        _rounded_normal(rng, 1994, 18, 1870, 2025, n_rows).astype(int), 1870, 2025
    )
    lot_size = _rounded_normal(rng, 7200, 3200, 1200, 40000, n_rows)
    crime_rate = np.round(np.clip(rng.beta(2.2, 5.0, n_rows), 0.01, 0.65), 3)
    school_rating = np.clip(_rounded_normal(rng, 6.4, 1.6, 1, 10, n_rows), 1, 10)
    distance_to_cbd = np.round(np.clip(rng.gamma(2.2, 4.5, n_rows), 0.5, 45), 2)

    year_built = np.minimum(year_built, 2025)
    age = 2025 - year_built

    price = 42_000.0
    price *= (area / 1000.0) ** 1.18
    price *= 1 + 0.075 * (bedrooms - 3)
    price *= 1 + 0.058 * (bathrooms - 2)
    price *= 0.968**age
    price *= 1 + 0.085 * (overall_quality - 5)
    price *= np.array([NEIGHBORHOODS[n] for n in neighborhood])
    price *= np.array([GARAGE_SPOTS[g] for g in garage])
    price *= np.array([FLOOR_MARGINAL[f] for f in floors])
    price *= np.array([RENOVATION[r] for r in renovation])
    price *= np.array([BASEMENT[b] for b in has_basement])
    price *= 1 + 0.13 * has_pool
    price *= 1 + 0.07 * air_conditioning
    price *= np.array([FURNISHED["Y" if f else "N"] for f in furnished])
    price *= np.array([MAIN_ROAD["Y" if m else "N"] for m in main_road])
    price *= 1 - 0.22 * crime_rate
    price *= 1 + 0.045 * (school_rating - 5)
    price *= np.exp(-0.026 * distance_to_cbd)
    price *= 1 + 0.012 * np.log1p(lot_size / 1000.0)
    price *= rng.lognormal(0.0, 0.085, n_rows)

    df = pd.DataFrame(
        {
            "area_sqft": area.astype(int),
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms,
            "age": age.astype(int),
            "floors": floors.astype(int),
            "garage": garage.astype(int),
            "lot_size_sqft": lot_size.astype(int),
            "has_basement": has_basement,
            "has_pool": has_pool,
            "air_conditioning": air_conditioning,
            "furnished": furnished,
            "main_road": main_road,
            "renovation": renovation,
            "neighborhood": neighborhood,
            "overall_quality": overall_quality.astype(int),
            "year_built": year_built,
            "crime_rate": crime_rate,
            "school_rating": school_rating.astype(int),
            "distance_to_cbd": distance_to_cbd,
            "price": np.round(price, 0),
        }
    )
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_dataframe()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows -> {OUT_PATH}")
    print(df.head())
    print()
    print(df["price"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
