import numpy as np
from scipy.stats import norm, t, skewnorm, gennorm, exponnorm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq


def standardize(rv, param):
    if type(rv) == str:
        rv = get_rv_from_name(rv)
    mean = rv.mean(param)
    std = rv.std(param)
    return rv(param, loc=-mean / std, scale=1 / std)


def kl_divergence(rv):
    def func(x):
        return rv.pdf(x) * (rv.logpdf(x) - norm.logpdf(x))

    return quad(func, -np.inf, np.inf)[0]


def wasserstein_distance(rv):
    def func(x):
        return np.abs(rv.cdf(x) - norm.cdf(x))

    return quad(func, -np.inf, np.inf)[0]


def target_function(rv, target, metric="wasserstein"):
    if metric == "wasserstein":

        def func(param):
            return wasserstein_distance(standardize(rv, param)) - target

    elif metric == "kl":

        def func(param):
            return kl_divergence(standardize(rv, param)) - target

    else:
        raise ValueError("metric must be wasserstein or kl")
    return func


def get_rv_from_name(rv_name):
    rv_dict = {
        "skewnorm": skewnorm,
        "exponnorm": exponnorm,
        "gennormsteep": gennorm,
        "gennormflat": gennorm,
        "t": t,
    }
    return rv_dict[rv_name]


def binary_search(rv_name, distance, metric):
    rv = get_rv_from_name(rv_name)
    func = target_function(rv, distance, metric=metric)
    if metric == "wasserstein":
        range_dict = {
            "skewnorm": (1e-4, 30.0),
            "exponnorm": (1e-4, 15.0),
            "gennormsteep": (1e-1, 2.0 - 1e-4),
            "gennormflat": (2.0 + 1e-4, 50.0),
            "t": (3.0, 200.0),
        }
    elif metric == "kl":
        range_dict = {
            "skewnorm": (1e-4, 30.0),
            "exponnorm": (1e-4, 15.0),
            "gennormsteep": (1e-1, 2.0 - 1e-4),
            "gennormflat": (2.0 + 1e-4, 200.0),
            "t": (3.0, 200.0),
        }
    return brentq(func, *range_dict[rv_name])


def standardized_rv_at_distance(distribution, distance, metric="wasserstein"):
    # if metric is kl, sensitive to adust range of binary search
    assert metric in ["wasserstein", "kl"], "metric must be wasserstein or kl"
    params_dict = {
        "skewnorm": {
            "0.03": 1.141679535895037,
            "0.06": 1.668027646656356,
            "0.09": 2.253555993158534,
            "0.12": 3.052977442461724,
            "0.15": 4.441693019739707,
        },
        "exponnorm": {
            "0.03": 0.5274333543184184,
            "0.06": 0.7361945074942922,
            "0.09": 0.9307079975424131,
            "0.12": 1.1365153042836023,
            "0.15": 1.372114598160624,
        },
        "gennormsteep": {
            "0.03": 1.685486347382175,
            "0.06": 1.446878209856004,
            "0.09": 1.2592111500311147,
            "0.12": 1.1075283854228473,
            "0.15": 0.9822742249929434,
        },
        "gennormflat": {
            "0.03": 2.4358709097539135,
            "0.06": 3.0868574329392504,
            "0.09": 4.188574703248306,
            "0.12": 6.60223527240027,
            "0.15": 23.021018170499307,
        },
        "t": {
            "0.03": 13.911718115376004,
            "0.06": 7.606345474941293,
            "0.09": 5.498186625845221,
            "0.12": 4.441398730633352,
            "0.15": 3.8067196925891835,
        },
    }
    if metric == "wasserstein" and distance in [0.03, 0.06, 0.09, 0.12, 0.15]:
        param = params_dict[distribution][str(distance)]
    else:
        param = binary_search(distribution, distance, metric=metric)

    rv = get_rv_from_name(distribution)
    return standardize(rv, param)


def plot_at_same_distance(
    noise_list,
    distance,
    metric="wasserstein",
    ylim=(-0.02, 0.75),
    is_save=False,
    is_title=True,
):
    plt.figure()
    x = np.linspace(-4, 4, 1000)
    for noise in noise_list:
        rv = standardized_rv_at_distance(noise, distance, metric=metric)
        plt.plot(x, rv.pdf(x), label=noise, lw=0.5)
    plt.plot(x, norm.pdf(x), label="norm", lw=0.8)
    plt.legend(frameon=False)
    if is_title:
        plt.title(f"Wasserstein Distance: {distance:.2f}")
    plt.ylim(*ylim)
    plt.xlabel("x")
    plt.ylabel("probability density")
    if is_save:
        plt.savefig(
            f"same_distance_{distance}.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.0,
        )
    plt.show()


def plot_family_of_distributions(noise, distances, metric="wasserstein", is_save=False):
    plt.figure()
    x = np.linspace(-4, 4, 1000)
    for dist in distances:
        rv = standardized_rv_at_distance(noise, dist, metric=metric)
        plt.plot(x, rv.pdf(x), label=f"dist: {dist:.2f}", lw=0.5)
    plt.plot(x, norm.pdf(x), label="dist: 0.00", lw=0.5)
    plt.legend(frameon=False)
    plt.title(f"Family of Distribution: {noise}")
    plt.ylim(-0.02, 0.75)
    plt.xlabel("x")
    plt.ylabel("probability density")
    if is_save:
        plt.savefig(
            f"family_of_distribution_{noise}.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.0,
        )
    plt.show()
