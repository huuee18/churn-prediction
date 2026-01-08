import numpy as np


# =====================================================
# 1. Fixed-length sliding window (TSMixer / LSTM / GRU)
# =====================================================
def create_fixed_window_sequences(
    df,
    features,
    target='CHURN',
    window_size=6
):
    """
    Dự đoán churn ở tháng kế tiếp (t+1)

    X: [t-window_size ... t-1]
    y: churn tại t
    """

    X, y = [], []

    for pol_num, group in df.groupby('POL_NUM'):
        group = group.sort_values('YEAR_MONTH_TRANS')

        X_vals = group[features].values
        y_vals = group[target].values

        # cần ít nhất window_size + 1 để predict future
        if len(group) <= window_size:
            continue

        for i in range(len(group) - window_size):
            X.append(X_vals[i:i + window_size])
            y.append(y_vals[i + window_size])

    return np.asarray(X), np.asarray(y)


# =====================================================
# 2. Early-warning window (predict before churn)
# =====================================================
def create_early_warning_sequences(
    df,
    features,
    target='CHURN',
    window_size=6,
    horizon=3
):
    """
    Dự đoán churn sau 'horizon' tháng
    (ví dụ early warning trước 3 tháng)

    X: [t-window_size ... t-1]
    y: churn tại t + horizon
    """

    X, y = [], []

    for pol_num, group in df.groupby('POL_NUM'):
        group = group.sort_values('YEAR_MONTH_TRANS')

        X_vals = group[features].values
        y_vals = group[target].values

        if len(group) <= window_size + horizon - 1:
            continue

        for i in range(len(group) - window_size - horizon + 1):
            X.append(X_vals[i:i + window_size])
            y.append(y_vals[i + window_size + horizon - 1])

    return np.asarray(X), np.asarray(y)


# =====================================================
# 3. Aggregate TS → Tabular (N-BEATS / DLinear)
# =====================================================
def create_aggregated_features(
    df,
    features,
    target='CHURN'
):
    """
    Chuyển time-series thành tabular
    Dùng cho N-BEATS classifier / DLinear

    Lưu ý:
    - label = churn ở tháng CUỐI
    - không dùng dữ liệu sau churn
    """

    agg_funcs = ['mean', 'std', 'min', 'max', 'last']

    X = (
        df.groupby('POL_NUM')[features]
        .agg(agg_funcs)
    )

    X.columns = [f"{c}_{f}" for c, f in X.columns]
    X = X.fillna(0)

    # churn tại thời điểm cuối
    y = (
        df.groupby('POL_NUM')[target]
        .last()
        .values
    )

    return X.values, y


# =====================================================
# 4. Utility: kiểm tra leakage nhanh
# =====================================================
def sanity_check_sequences(X, y):
    """
    Kiểm tra nhanh:
    - shape hợp lý
    - không toàn 0 / toàn 1
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Churn ratio:", np.mean(y))

