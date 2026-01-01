import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def preprocess_data(df):
    df = df.sort_values(['POL_NUM', 'YEAR_MONTH_TRANS'])

    # =====================
    # 1. Feature engineering (SAFE)
    # =====================
    df['CLAIM_RATIO'] = df['CLAIM_AMT'] / (df['PREM_AMT'] + 1)

    # clip ratio (rất quan trọng)
    df['CLAIM_RATIO'] = df['CLAIM_RATIO'].clip(0, 5)

    cols_to_log = ['PREM', 'PREM_AMT', 'POS_AMT', 'CLAIM_AMT']
    for col in cols_to_log:
        df[f'{col}_LOG'] = np.log1p(df[col].clip(0))

    df['HAS_CLAIM'] = (df['CLAIM_COUNT'] > 0).astype(int)
    df['HAS_POS_AMT'] = (df['POS_AMT'] > 0).astype(int)

    # =====================
    # 2. PAY_FREQ (business-safe)
    # =====================
    freq_map = {1:12, 2:4, 3:2, 4:1, 5:1, 6:0}
    df['PAY_FREQ_NUM'] = df['PAY_FREQ_TYPE'].map(freq_map)
    df['IS_SINGLE_PAY'] = (df['PAY_FREQ_NUM'] <= 1).astype(int)

    # =====================
    # 3. Lag & Rolling (ANTI-LEAKAGE)
    # =====================
    lag_features = ['PREM_AMT', 'POS_AMT', 'CLAIM_AMT']
    for col in lag_features:
        df[f'{col}_LAG_1'] = df.groupby('POL_NUM')[col].shift(1)
        df[f'{col}_ROLL_MEAN_3'] = (
            df.groupby('POL_NUM')[col]
              .rolling(3, min_periods=2)
              .mean()
              .reset_index(0, drop=True)
        )

        # trend thay vì absolute value
        df[f'{col}_DELTA_1'] = df[col] - df[f'{col}_LAG_1']

    # flag tháng đầu (thay vì fill 0)
    df['IS_FIRST_MONTH'] = (
        df.groupby('POL_NUM').cumcount() == 0
    ).astype(int)

    # =====================
    # 4. Categorical (SAFE cho deep model)
    # =====================
    insur_freq = df['INSUR_TYPE'].value_counts(normalize=True)
    df['INSUR_TYPE_FREQ'] = df['INSUR_TYPE'].map(insur_freq)

    # =====================
    # 5. Missing handling (KHÔNG fill 0 bừa)
    # =====================
    for col in df.columns:
        if col != 'CHURN' and df[col].isna().any():
            df[f'{col}_MISSING'] = df[col].isna().astype(int)

    df = df.fillna(df.median(numeric_only=True))

    # =====================
    # 6. Noise injection (giảm overfit cho deep TS)
    # =====================
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('CHURN')

    noise = np.random.normal(0, 0.01, size=df[num_cols].shape)
    df[num_cols] = df[num_cols] + noise

    return df, num_cols



