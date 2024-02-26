import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import json

import fire
import matplotlib.pyplot as plt
import numpy as np
import scipy
import si4dnn as si
import tensorflow as tf
from mpmath import mp
from si4dnn.si import NoHypothesisError
from tqdm import tqdm

import si4vae
import wandb
from si4vae.model import SamplingLayer
from util import (ar1_image_covariance, multivariate_normal_sample,
                  nongaussian_sample)


def compute_group_statistics(data, k):
    # Divide data into k groups
    n = len(data)
    groups = np.array_split(data, k)

    # Compute the mean of each group
    group_means = np.array([np.mean(group) for group in groups])

    # Compute the overall mean of the group means
    overall_mean = np.mean(group_means)

    # Compute the standard error of the group means
    standard_error = np.std(group_means, ddof=1) / np.sqrt(k)

    return overall_mean, standard_error


def fn(
    args: tuple,
):
    X = args[0]
    var = args[1]
    model_path = args[2]
    smoothing = args[3]
    shape = args[4]
    thr = args[5]
    signal = args[6]
    parametric_mode = args[7]
    alpha = args[8]

    model = tf.keras.models.load_model(
        model_path, custom_objects={"SamplingLayer": SamplingLayer}
    )

    si_ae = si4vae.si.AEInferenceNorm(model, thr, var=var, smoothing=smoothing)

    start_time = time.time()
    if parametric_mode == "exhaustive":
        p_result = si_ae.inference(X, exhaustive=True)
    elif parametric_mode == "decision":
        p_result = si_ae.inference(X, termination_criterion="decision",significance_level=alpha)
    elif parametric_mode == "precision":
        p_result = si_ae.inference(X, termination_criterion="precision")
    else:
        assert False, "parametric_mode must be exhaustive, decision or precision"
    end_time = time.time()

    oc_p_value = si_ae.inference(X, over_conditioning=True).p_value
    naive_p_value = si_ae.naive_inference(X)
    bonf_p_value = si_ae.bonf_inference(X)

    return p_result, oc_p_value, naive_p_value, bonf_p_value, end_time - start_time


def experiment(
    model_path: str,
    size: int,
    thr: float,
    signal: float | None = None,
    alpha: float = 0.05,
    smoothing: str | None = None,
    number_of_worker: int = 1,
    number_of_iter: int = 10,
    seed: int = 1234,
    parametric_mode: str = "exhaustive",
    cov: str | None = None,
    rho: float = 0.5,
    noise_distribution: str | None = None,
    ws_distance: float | None = None,
    **kwargs,
):
    """Experiment function for Selective Inference of Anomaly Detection using Auto Encoder

    Args:
        model_path(str) : path for trained tensorflow/keras model
        size(int) : width and height of input image
        thr(float) : threshold for reconstruction error in anomaly detection
        signal(float): signal for the input image
        alpha(float): significance level
        smoothing: smoothing method for the anomaly detection
        number_of_worker(int) : 
        number_of_iter(int) :
        seed(int) : Seed for np.random.randn and other sampling 
        parametric_mode(str) : Parametric Programming
        cov(str) : The structure of covariance matrix
        rho(float) : The strength of covariance
        noise_distribution(str) : Noise distribution
        ws_distance: How the noise distribution deviate from the standard gaussian N(0,1)
        **kwargs: keyword arguments for SelectiveInferenceNorm.inference
    Return:
        void
    """

    shape = (size, size, 1)
    np.random.seed(seed)

    if parametric_mode not in ["exhaustive", "decision", "precision"]:
        raise ValueError("parametric_mode must be exhaustive, decision or precision")

    config_dict = {
        "model_path": model_path,
        "size": size,
        "signal": signal,
        "thr": thr,
        "alpha": alpha,
        "number_of_iter": number_of_iter,
        "filter": smoothing,
        "seed": seed,
        "parametric_mode": parametric_mode,
        "cov": cov,
        "rho": rho,
        "noise_distribution": noise_distribution,
        "ws_distance": ws_distance,
        **kwargs,
    }

    print("Settings are")
    # print(config_dict)

    if noise_distribution is not None:
        def gen_data():
            X = nongaussian_sample(noise_distribution, ws_distance, 1, shape)

            if signal is not None:
                x_anker = np.random.randint(0, (size // 3) + 1)
                y_anker = np.random.randint(0, (size / 3) + 1)
                X[
                    0,
                    x_anker : x_anker + (size // 3),
                    y_anker : y_anker + (size // 3),
                    0,
                ] += signal

            return X

        var = 1
    elif cov is None:
        def gen_data():
            X = np.random.randn(1, *shape)
            if signal is not None:
                x_anker = np.random.randint(0, (size // 3) + 1)
                y_anker = np.random.randint(0, (size / 3) + 1)
                X[
                    0,
                    x_anker : x_anker + (size // 3),
                    y_anker : y_anker + (size // 3),
                    0,
                ] += signal

            return X

        var = 1

    elif cov == "ar1":
        var = ar1_image_covariance(size, size, rho)

        def gen_data():
            X = multivariate_normal_sample(var, 1, shape)
            if signal is not None:
                x_anker = np.random.randint(0, (size // 3) + 1)
                y_anker = np.random.randint(0, (size / 3) + 1)
                X[
                    0,
                    x_anker : x_anker + (size // 3),
                    y_anker : y_anker + (size // 3),
                    0,
                ] += signal
            return X

    else:
        assert False, "cov must be None or ar1"

    args = (
        (
            gen_data(),
            var,
            model_path,
            smoothing,
            shape,
            thr,
            signal,
            parametric_mode,
            alpha,
        )
        for i in iter(int, 1)
    )

    # results = list(tqdm(executor.map(fn, args), total=number_of_iter))
    with tqdm(total=number_of_iter) as pbar:
        with ProcessPoolExecutor(max_workers=number_of_worker) as executor:
            number_of_completed_jobs = 0
            number_of_failed_jobs = 0
            futures = []
            results = []
            while number_of_completed_jobs < number_of_iter:
                for i in range(number_of_worker):
                    future = executor.submit(fn, next(args))
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        number_of_completed_jobs += 1
                        pbar.update(1)
                    except NoHypothesisError:
                        number_of_failed_jobs += 1
                        if number_of_failed_jobs % 50 == 0:
                            print("number_of_failed_jobs:", number_of_failed_jobs)
                        continue

                    futures.remove(future)

                    if number_of_completed_jobs >= number_of_iter:
                        break

                    futures.append(executor.submit(fn, next(args)))

        p_results = [result[0] for result in results if result is not None]
        oc_p_values = np.array([result[1] for result in results if result is not None])
        naive_p_values = np.array(
            [result[2] for result in results if result is not None]
        )
        bonf_p_values = np.array(
            [result[3] for result in results if result is not None]
        )
        times = np.array([result[4] for result in results if result is not None])

    p_values = np.array([p_result.p_value for p_result in p_results])
    p_reject_or_not_values = np.array(
        [p_result.sup_p <= alpha for p_result in p_results]
    )
    search_count = np.array([p_result.search_count for p_result in p_results])

    ks_p_value = scipy.stats.kstest(p_values, "uniform")[1]
    ks_oc_p_value = scipy.stats.kstest(oc_p_values, "uniform")[1]
    ks_naive_p_value = scipy.stats.kstest(naive_p_values, "uniform")[1]
    ks_bonf_p_value = scipy.stats.kstest(bonf_p_values, "uniform")[1]

    p_value_mean, p_value_standard_error = compute_group_statistics(
        p_reject_or_not_values, 10
    )
    oc_p_value_mean, oc_p_value_standard_error = compute_group_statistics(
        oc_p_values <= alpha, 10
    )
    naive_p_value_mean, naive_p_value_standard_error = compute_group_statistics(
        naive_p_values <= alpha, 10
    )
    bonf_p_value_mean, bonf_p_value_standard_error = compute_group_statistics(
        bonf_p_values <= alpha, 10
    )

    result_dict = {
        "VAE-AD Test": np.mean(p_reject_or_not_values),
        "OC": np.mean(oc_p_values < alpha),
        "Naive": np.mean(naive_p_values < alpha),
        "Bonf": np.mean(bonf_p_values < alpha),
        "p-value-mean": p_value_mean,
        "p-value-se": p_value_standard_error,
        "oc-p-value-mean": oc_p_value_mean,
        "oc-p-value-se": oc_p_value_standard_error,
        "naive-p-value-mean": naive_p_value_mean,
        "naive-p-value-se": naive_p_value_standard_error,
        "bonf-p-value-mean": bonf_p_value_mean,
        "bonf-p-value-se": bonf_p_value_standard_error,
        "ks-p": ks_p_value,
        "ks-oc-p": ks_oc_p_value,
        "ks-naive-p": ks_naive_p_value,
        "ks-bonf-p": ks_bonf_p_value,
        "number-of-error": number_of_iter - len(p_values),
        "search-count": np.mean(search_count),
        "time": np.mean(times),
    }

    print("========== CONFIG ==========")
    print(config_dict)
    print("")
    print("========== RESULT ==========")
    print(result_dict)

    if signal is None:
        experiment = "typeIerror"
        signal = 0.0
    else :
        experiment = "power"
    
    if cov is not None:
        noise = "independent"
    else :
        noise = "correlated"
    
    if noise_distribution is None:
        noise = noise_distribution

    with open(f"experimental_result/{experiment}_{noise}_{size}_{signal}.csv", "w") as json_file:
        json.dump(result_dict, json_file)

if __name__ == "__main__":
    fire.Fire(experiment)
