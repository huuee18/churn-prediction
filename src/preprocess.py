import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    FINAL VERSION
    Optimized preprocessing for churn prediction
    - Time-series safe
    - Leakage-safe feature engineering
    - Strong churn behavior signals
    - Suitable for TSMixer / LSTM / GRU / XGBoost
    """

    # ==========================================
    # 0. SORT TIME ORDER
    # ==========================================
    df = df.sort_values(
        ["POL_NUM", "YEAR_MONTH_TRANS"]
    ).copy()

    # ==========================================
    # 1. BASIC CLEANING
    # ==========================================
    numeric_base_cols = [
        "PREM",
        "PREM_AMT",
        "POS_AMT",
        "CLAIM_AMT",
        "CLAIM_COUNT"
    ]

    for col in numeric_base_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    # ==========================================
    # 2. CORE FEATURE ENGINEERING
    # ==========================================
    df["CLAIM_RATIO"] = (
        df["CLAIM_AMT"] /
        (df["PREM_AMT"] + 1)
    ).clip(0, 5)

    log_cols = [
        "PREM",
        "PREM_AMT",
        "POS_AMT",
        "CLAIM_AMT"
    ]

    for col in log_cols:
        df[f"{col}_LOG"] = np.log1p(
            df[col].clip(lower=0)
        )

    df["HAS_CLAIM"] = (
        df["CLAIM_COUNT"] > 0
    ).astype(int)

    df["HAS_POS_AMT"] = (
        df["POS_AMT"] > 0
    ).astype(int)

    # ==========================================
    # 3. PAYMENT BEHAVIOR FEATURES
    # ==========================================
    freq_map = {
        1: 12,
        2: 4,
        3: 2,
        4: 1,
        5: 1,
        6: 0
    }

    df["PAY_FREQ_NUM"] = df[
        "PAY_FREQ_TYPE"
    ].map(freq_map).fillna(0)

    df["IS_SINGLE_PAY"] = (
        df["PAY_FREQ_NUM"] <= 1
    ).astype(int)

    # ==========================================
    # 4. TIME SERIES FEATURES
    # ==========================================
    lag_cols = [
        "PREM_AMT",
        "POS_AMT",
        "CLAIM_AMT"
    ]

    for col in lag_cols:

        grp = df.groupby("POL_NUM")[col]

        # lag
        df[f"{col}_LAG1"] = grp.shift(1)
        df[f"{col}_LAG2"] = grp.shift(2)
        df[f"{col}_LAG3"] = grp.shift(3)

        # rolling mean
        df[f"{col}_ROLL3"] = (
            grp.shift(1)
               .groupby(df["POL_NUM"])
               .rolling(3, min_periods=1)
               .mean()
               .reset_index(level=0, drop=True)
        )

        # rolling std
        df[f"{col}_STD3"] = (
            grp.shift(1)
               .groupby(df["POL_NUM"])
               .rolling(3, min_periods=2)
               .std()
               .reset_index(level=0, drop=True)
        )

        # delta
        df[f"{col}_DELTA1"] = (
            df[col] - df[f"{col}_LAG1"]
        )

        # decline
        df[f"{col}_DECLINE_FLAG"] = (
            df[col] < df[f"{col}_LAG1"]
        ).astype(int)

    # ==========================================
    # 5. CHURN-SPECIFIC FEATURES
    # ==========================================

    # premium = 0
    df["ZERO_PREM_FLAG"] = (
        df["PREM_AMT"] == 0
    ).astype(int)

    # zero premium last 3 months
    df["ZERO_PREM_3M"] = (
        df.groupby("POL_NUM")["ZERO_PREM_FLAG"]
          .shift(1)
          .groupby(df["POL_NUM"])
          .rolling(3, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    # contract age
    df["POLICY_AGE_MONTH"] = (
        df.groupby("POL_NUM")
          .cumcount() + 1
    )

    # first month flag
    df["IS_FIRST_MONTH"] = (
        df["POLICY_AGE_MONTH"] == 1
    ).astype(int)

    # premium decline streak (very useful)
    df["PREM_DECLINE_STREAK"] = (
        df.groupby("POL_NUM")["PREM_AMT_DECLINE_FLAG"]
          .rolling(3, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    # months since claim
    df["CLAIM_EVENT"] = (
        df["CLAIM_COUNT"] > 0
    ).astype(int)

    df["MONTHS_SINCE_CLAIM"] = (
        df.groupby("POL_NUM")["CLAIM_EVENT"]
          .cumsum()
    )

    # ==========================================
    # 6. CATEGORICAL ENCODING
    # ==========================================
    if "INSUR_TYPE" in df.columns:

        insur_freq = df["INSUR_TYPE"].value_counts(
            normalize=True
        )

        df["INSUR_TYPE_FREQ"] = df[
            "INSUR_TYPE"
        ].map(insur_freq)

    # ==========================================
    # 7. MISSING FLAGS
    # ==========================================
    for col in df.columns:

        if col != "CHURN" and df[col].isna().any():

            df[f"{col}_MISS"] = (
                df[col].isna().astype(int)
            )

    # ==========================================
    # 8. FILL MISSING
    # ==========================================
    num_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    remove_cols = [
        "CHURN",
        "POL_NUM",
        "YEAR_MONTH_TRANS"
    ]

    num_cols = [
        c for c in num_cols
        if c not in remove_cols
    ]

    for col in num_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # ==========================================
    # 9. FINAL CLEANUP
    # ==========================================
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    for col in num_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    return df, num_cols
