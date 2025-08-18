#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MovieLens 100K → HDF5 with leakage-aware preprocessing.

Features:
- Stable merges and dtypes
- Gender (0/1), Age scaling (zscore|minmax|raw)
- Occupation: one-hot (default) or integer IDs
- Optional gender-occupation rebalancing (undersample) to reduce leakage
- Correlation report (Cramér’s V) + warnings
- Writes mapping info and preprocessing metadata to HDF5 attrs

Usage: just run this file in the repo root (expects ml-100k/ present)
"""

import os
import h5py
import numpy as np
import pandas as pd
from collections import Counter

# =========================
# CONFIG
# =========================
BASE_DIR = "ml-100k"
OUTPUT_FILE = "movielens_100k_with_attrs.hdf5"

# Core toggles
AGE_SCALING = "zscore"      # "zscore" | "minmax" | "raw"
OCC_ENCODING = "id"         # "onehot" | "id"
INCLUDE_GENDER = True
INCLUDE_AGE = True
INCLUDE_OCC = True

# Auditor-friendly options
DROP_IDS = False            # If True, do not store user/movie IDs in HDF5
REBALANCE_OCC_BY_GENDER = False   # If True, undersample so occ distribution per gender is balanced

# Correlation warning threshold (Cramér’s V)
CRAMERS_V_WARN = 0.25

# Repro
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================
# Helpers
# =========================

def cramers_v(cat_a: pd.Series, cat_b: pd.Series) -> float:
    """Cramér’s V for two categorical series."""
    contingency = pd.crosstab(cat_a, cat_b)
    chi2 = ((contingency - contingency.values.sum() * np.outer(contingency.sum(1), contingency.sum(0)) / (contingency.values.sum()**2))**2
            / (np.outer(contingency.sum(1), contingency.sum(0)) / contingency.values.sum())).to_numpy().sum()
    n = contingency.values.sum()
    r, k = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-12)))

def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def minmax01(x: pd.Series) -> pd.Series:
    x_min, x_max = x.min(), x.max()
    denom = (x_max - x_min) if (x_max > x_min) else 1.0
    return (x - x_min) / denom

def rebalance_by_gender_occupation(df: pd.DataFrame, gender_col: str, occ_col: str) -> pd.DataFrame:
    """
    Undersample so that for each occupation, M/F counts match the min count for that occ.
    Keeps timestamp ordering random within each cell.
    """
    parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for occ, df_occ in df.groupby(occ_col):
        g_counts = df_occ[gender_col].value_counts()
        if len(g_counts) < 2:
            # Only one gender present; skip (no balancing possible)
            parts.append(df_occ)
            continue
        m = int(g_counts.min())
        for g in [0, 1]:
            sub = df_occ[df_occ[gender_col] == g]
            if len(sub) > m:
                idx = rng.choice(sub.index.values, size=m, replace=False)
                parts.append(sub.loc[idx])
            else:
                parts.append(sub)
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return out

# =========================
# 1) Load
# =========================
ratings = pd.read_csv(
    os.path.join(BASE_DIR, "u.data"),
    sep="\t", names=["userId", "movieId", "rating", "timestamp"],
    dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
)
users = pd.read_csv(
    os.path.join(BASE_DIR, "u.user"),
    sep="|", names=["userId", "age", "gender", "occupation", "zip"],
    dtype={"userId": np.int32, "age": np.int32, "gender": str, "occupation": str, "zip": str},
)
movies = pd.read_csv(
    os.path.join(BASE_DIR, "u.item"),
    sep="|", encoding="latin-1",
    names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + [f"genre_{i}" for i in range(19)],
    usecols=list(range(24)),
    dtype={"movieId": np.int32, **{f"genre_{i}": np.uint8 for i in range(19)}}
)

# =========================
# 2) Merge
# =========================
df = ratings.merge(users, on="userId", how="inner").merge(movies, on="movieId", how="inner")

# =========================
# 3) Basic transforms
# =========================
# Genres: already 0/1
genre_cols = [f"genre_{i}" for i in range(19)]
genre_matrix = df[genre_cols].values.astype(np.uint8)

# Gender -> 0/1
if INCLUDE_GENDER:
    df["gender_encoded"] = (df["gender"].str.upper() == "M").astype(np.uint8)

# Age scaling
if INCLUDE_AGE:
    if AGE_SCALING == "zscore":
        df["age_scaled"] = zscore(df["age"]).astype(np.float32)
        age_meta = {"mode": "zscore", "mean": float(df["age"].mean()), "std": float(df["age"].std(ddof=0))}
    elif AGE_SCALING == "minmax":
        df["age_scaled"] = minmax01(df["age"]).astype(np.float32)
        age_meta = {"mode": "minmax", "min": float(df["age"].min()), "max": float(df["age"].max())}
    elif AGE_SCALING == "raw":
        df["age_scaled"] = df["age"].astype(np.float32)
        age_meta = {"mode": "raw"}
    else:
        raise ValueError(f"Unknown AGE_SCALING: {AGE_SCALING}")
else:
    age_meta = {"mode": "excluded"}

# Occupation encode
occ_map = None
if INCLUDE_OCC:
    if OCC_ENCODING == "onehot":
        # This branch is no longer used since OCC_ENCODING is set to "id"
        pass
    elif OCC_ENCODING == "id":
        uniq = sorted(df["occupation"].unique())
        occ_map = {occ: i for i, occ in enumerate(uniq)}
        df["occupation_id"] = df["occupation"].map(occ_map).astype(np.int32)
    else:
        raise ValueError(f"Unknown OCC_ENCODING: {OCC_ENCODING}")

# =========================
# 4) Correlation report (and optional rebalance)
# =========================
if INCLUDE_GENDER and INCLUDE_OCC:
    # Build categorical occupation (id for report)
    if occ_map is None:
        uniq = sorted(df["occupation"].unique())
        occ_map = {occ: i for i, occ in enumerate(uniq)}
    occ_id_report = df["occupation"].map(occ_map).astype(np.int32)

    V = cramers_v(df["gender_encoded"], occ_id_report)
    print(f"[Report] Cramér’s V between gender and occupation: {V:.3f}")
    if V >= CRAMERS_V_WARN:
        print(f"[Warning] Strong gender↔occupation association (≥ {CRAMERS_V_WARN}). "
              "Auditors may trivially infer gender from occupation. Consider one-hot or rebalancing.")

    if REBALANCE_OCC_BY_GENDER:
        before = (df.groupby(["gender_encoded", "occupation"]).size()
                    .unstack(0, fill_value=0).head(5))
        print("[Before rebalance] head of counts (occupation rows × gender cols):")
        print(before)
        df = rebalance_by_gender_occupation(df, "gender_encoded", "occupation")
        print("[After rebalance] sample size:", len(df))

# Recompute genre matrix in case of rebalance
genre_matrix = df[genre_cols].values.astype(np.uint8)

# =========================
# 5) Arrays
# =========================
user_ids = df["userId"].astype(np.int32).values
movie_ids = df["movieId"].astype(np.int32).values
ratings_arr = df["rating"].astype(np.float32).values

# =========================
# 6) Write HDF5
# =========================
with h5py.File(OUTPUT_FILE, "w") as f:
    # Core
    if not DROP_IDS:
        f.create_dataset("user_ids", data=user_ids)
        f.create_dataset("item_ids", data=movie_ids)
    f.create_dataset("ratings", data=ratings_arr)
    f.create_dataset("genre_onehot", data=genre_matrix)

    # Gender
    if INCLUDE_GENDER:
        f.create_dataset("gender", data=df["gender_encoded"].astype(np.uint8).values)

    # Age
    if INCLUDE_AGE:
        f.create_dataset("age", data=df["age_scaled"].astype(np.float32).values)
        f.create_dataset("age_raw", data=df["age"].astype(np.int32).values)  # always store raw age

    # Occupation (always id encoding)
    if INCLUDE_OCC:
        f.create_dataset("occupation", data=df["occupation_id"].astype(np.int32).values)
        # store string mapping
        keys = np.array(list(occ_map.keys()), dtype="S")
        vals = np.array(list(occ_map.values()), dtype=np.int32)
        f.create_dataset("occupation_str_keys", data=keys)
        f.create_dataset("occupation_int_vals", data=vals)

    # Metadata
    f.attrs["global_mean"] = float(np.mean(ratings_arr))
    f.attrs["num_genres"] = 19
    f.attrs["age_meta"] = str(age_meta)
    f.attrs["occupation_encoding"] = OCC_ENCODING
    f.attrs["include_gender"] = int(INCLUDE_GENDER)
    f.attrs["include_age"] = int(INCLUDE_AGE)
    f.attrs["include_occupation"] = int(INCLUDE_OCC)
    f.attrs["drop_ids"] = int(DROP_IDS)
    f.attrs["rebalance_occ_by_gender"] = int(REBALANCE_OCC_BY_GENDER)
    f.attrs["random_seed"] = RANDOM_SEED

    # Add genre_labels attribute with standard ML-100K genre names
    genre_labels = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    f.attrs["genre_labels"] = np.array(genre_labels, dtype="S")

print(f"✅ Saved: {OUTPUT_FILE}")