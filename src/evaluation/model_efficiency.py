# -*- coding: utf-8 -*-
import time
import psutil
import os
import pandas as pd


def measure_training_time(model, X_train, y_train, epochs=10, batch_size=32):

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    end = time.perf_counter()

    return end - start


def measure_inference_time(model, X_test):

    start = time.perf_counter()

    y_pred = model.predict(X_test, verbose=0)

    end = time.perf_counter()

    total_time = end - start
    per_sample = total_time / len(X_test)

    return total_time, per_sample


def count_parameters(model):

    return model.count_params()


def estimate_model_size(model):

    params = model.count_params()

    # float32 = 4 bytes
    size_mb = params * 4 / (1024 ** 2)

    return size_mb


def measure_memory_usage(model, X_sample):

    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / (1024 ** 2)

    model.predict(X_sample, verbose=0)

    mem_after = process.memory_info().rss / (1024 ** 2)

    return mem_after - mem_before


def benchmark_efficiency(models, X_test):

    rows = []

    X_sample = X_test[:200]

    for name, model in models.items():

        print(f"⚙️ Measuring efficiency: {name}")

        total_inf, per_sample = measure_inference_time(model, X_test)

        params = count_parameters(model)

        size_mb = estimate_model_size(model)

        mem = measure_memory_usage(model, X_sample)

        rows.append({
            "Model": name,
            "Parameters": params,
            "Approx_Model_Size_MB": size_mb,
            "Inference_Time_Total_sec": total_inf,
            "Inference_Time_Per_Sample_ms": per_sample * 1000,
            "Memory_Usage_MB": mem
        })

    return pd.DataFrame(rows)
