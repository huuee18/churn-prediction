import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_fixed_window_sequences(
    df,
    features,
    target='CHURN',
    window_size=6
):
    """
    Fixed-length sliding window sequences
    Phù hợp TSMixer / Transformer-like models
    """
    X, y = [], []

    for pol_num, group in df.groupby('POL_NUM'):
        group = group.sort_values('YEAR_MONTH_TRANS')

        values = group[features].values
        labels = group[target].values

        if len(values) < window_size:
            continue

        for i in range(len(values) - window_size + 1):
            X.append(values[i:i+window_size])
            y.append(labels[i+window_size-1])

    return np.array(X), np.array(y)
def create_aggregated_features(
    df,
    features,
    target='CHURN'
):
    """
    Aggregate time-series → tabular
    Phù hợp N-BEATS classifier / DLinear
    """
    agg_funcs = ['mean', 'std', 'min', 'max', 'last']

    X = (
        df.groupby('POL_NUM')[features]
        .agg(agg_funcs)
        .fillna(0)
    )
    X.columns = [f"{c}_{f}" for c, f in X.columns]

    y = df.groupby('POL_NUM')[target].last()

    return X.values, y.values
