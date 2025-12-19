import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def preprocess_data(df):
    df = df.sort_values(['POL_NUM', 'YEAR_MONTH_TRANS'])

    # -------- Feature engineering --------
    df['CLAIM_RATIO'] = df['CLAIM_AMT'] / df['PREM_AMT'].replace(0, np.nan)

    cols_to_log = ['PREM', 'PREM_AMT', 'POS_AMT', 'CLAIM_AMT', 'TOTAL_RIDER_PREM']
    for col in cols_to_log:
        df[f'{col}_LOG'] = np.log1p(df[col])

    df['HAS_CLAIM'] = (df['CLAIM_COUNT'] > 0).astype(int)
    df['HAS_ROLE'] = (df['NUM_ROLES'] > 0).astype(int)
    df['HAS_POS_AMT'] = (df['POS_AMT'] > 0).astype(int)

    freq_map = {1:12, 2:4, 3:2, 4:1, 5:1, 6:0}
    df['PAY_FREQ_NUM'] = df['PAY_FREQ_TYPE'].map(freq_map)

    # -------- Lag & rolling --------
    lag_features = ['PREM', 'PREM_AMT', 'POS_AMT', 'CLAIM_AMT']
    for col in lag_features:
        df[f'{col}_LAG_1'] = df.groupby('POL_NUM')[col].shift(1)
        df[f'{col}_ROLL_MEAN_3'] = df.groupby('POL_NUM')[col].rolling(3).mean().reset_index(0, drop=True)

    # -------- Encoding --------
    df['INSUR_TYPE_LE'] = LabelEncoder().fit_transform(df['INSUR_TYPE'])

    # One-hot only for linear models
    df_linear = pd.get_dummies(df, columns=['INSUR_TYPE', 'PAY_FREQ_TYPE'])

    # -------- Scaling --------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('CHURN')

    scaler_tree = MinMaxScaler()
    df[num_cols] = scaler_tree.fit_transform(df[num_cols])

    scaler_linear = StandardScaler()
    df_linear[num_cols] = scaler_linear.fit_transform(df_linear[num_cols])

    return df, df_linear, num_cols

