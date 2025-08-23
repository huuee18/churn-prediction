import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def preprocess_data(df):
    """Tiền xử lý dữ liệu: feature engineering, log-transform, encoding, scaling"""
    # Tính CLAIM_RATIO
    df['CLAIM_RATIO'] = df['CLAIM_AMT'] / df['PREM_AMT'].replace(0, np.nan)
    
    # Log transform
    cols_to_log = ['PREM', 'PREM_AMT', 'POS_AMT', 'CLAIM_AMT', 'TOTAL_RIDER_PREM']
    for col in cols_to_log:
        df[f'{col}_LOG'] = np.log1p(df[col])
    
    # Feature nhị phân
    df['HAS_CLAIM'] = (df['CLAIM_COUNT'] > 0).astype(int)
    df['HAS_ROLE'] = (df['NUM_ROLES'] > 0).astype(int)
    df['HAS_POS_AMT'] = (df['POS_AMT'] > 0).astype(int)

    # PAY_FREQ_TYPE → số lần đóng/năm
    freq_map = {1:12, 2:4, 3:2, 4:1, 5:1, 6:0}
    df['PAY_FREQ_NUM'] = df['PAY_FREQ_TYPE'].map(freq_map)

    # Label encode
    le_insur = LabelEncoder()
    le_payfreq = LabelEncoder()
    df['INSUR_TYPE_LE'] = le_insur.fit_transform(df['INSUR_TYPE'])
    df['PAY_FREQ_TYPE_LE'] = le_payfreq.fit_transform(df['PAY_FREQ_TYPE'])

    # One-hot encode cho Logistic Regression
    df_linear = pd.get_dummies(df, columns=['INSUR_TYPE', 'PAY_FREQ_TYPE'])

    # Chuẩn hóa
    num_cols = [
        'POL_PERIOD', 'FACE_AMT', 'PREM', 'PAY_TIMES','TIMES_VALID_PAY',
        'POL_AGE','DAYS_SINCE_LAST_PAY','RIDER_COUNT','TOTAL_RIDER_PREM',
        'CUSTOMER_AGE','CLAIM_COUNT','PREM_AMT','POS_AMT','NUM_ROLES',
        'CLAIM_RATIO'
    ] + [f'{col}_LOG' for col in cols_to_log]

    scaler_tree = MinMaxScaler()
    df[num_cols] = scaler_tree.fit_transform(df[num_cols])

    scaler_linear = StandardScaler()
    df_linear[num_cols] = scaler_linear.fit_transform(df_linear[num_cols])

    return df, df_linear, num_cols
