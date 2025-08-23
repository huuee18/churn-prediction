import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_sequences(df, features, target='CHURN'):
    """Tạo sequence cho LSTM từ POL_NUM"""
    sequence_data = []
    for pol_num, group in df.groupby('POL_NUM'):
        group = group.sort_values('YEAR_MONTH_TRANS')
        X_seq = group[features].values
        y_label = group[target].iloc[-1]
        sequence_data.append((X_seq, y_label))

    maxlen = max(len(seq) for seq, _ in sequence_data)
    X = [x for x, _ in sequence_data]
    y = [y for _, y in sequence_data]
    X_pad = pad_sequences(X, dtype='float32', padding='post', maxlen=maxlen)
    return X_pad, np.array(y), maxlen
