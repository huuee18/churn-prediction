import numpy as np

def create_fixed_window_sequences(df, features, target, window_size):
    """
    Create sliding window sequences + return matching time index

    Returns:
        X_seq, y_seq, seq_time_index
    """

    data = df.sort_values(["POL_NUM", "YEAR_MONTH_TRANS"]).reset_index(drop=True)

    X, y = [], []
    seq_time_index = []

    grouped = data.groupby("POL_NUM")

    for _, group in grouped:
        group = group.reset_index(drop=True)

        values = group[features].values
        target_values = group[target].values
        times = group["YEAR_MONTH_TRANS"].values

        for i in range(len(group) - window_size):
            X.append(values[i : i + window_size])
            y.append(target_values[i + window_size])

            # 👉 Gán thời điểm của sample = thời điểm cuối window
            seq_time_index.append(times[i + window_size])

    X_seq = np.array(X)
    y_seq = np.array(y)
    seq_time_index = np.array(seq_time_index)

    return X_seq, y_seq, seq_time_index
