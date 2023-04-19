import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl


def fit_silent_period_to_power_distribution(spike_events, plot=False):
    df_silent_time = get_silent_time_distribution(spike_events)

    X = df_silent_time.index
    y = df_silent_time["silent_time"]

    popt, pcov = curve_fit(func_powerlaw, X, y)
    perr = np.sqrt(np.diag(pcov))
    popt = np.round(popt, 2)

    if plot:
        plt.plot(X, func_powerlaw(X, *popt))

    print("Function to fit: c * x ** (-m)")

    print(
        f"Curve fit: c={popt[0]} +- {round(perr[0], 3)}, m={popt[1]} +- {round(perr[1], 3)}"
    )
    print(f"Parameter error: {perr}")

    return popt[:2]


def plot_silent_period_distribution(spike_events):
    df_silent_time = get_silent_time_distribution(spike_events)

    df_silent_time.reset_index().plot(
        logx=True,
        logy=True,
        kind="scatter",
        x="index",
        y="silent_time",
        title="Silent time distribution, best power law fit",
    )


def func_powerlaw(x, c, m):
    return c * x ** (-m)


def get_silent_time_distribution(spike_events):
    df_spike_events = (
        pd.DataFrame(spike_events).sort_values("times").reset_index(drop=True)
    )
    df_spike_events["times"] = df_spike_events["times"] * 10

    df_silent_time = (
        df_spike_events.assign(
            silent_time=df_spike_events["times"]
            - df_spike_events["times"].shift(periods=1)
        )
        .query("silent_time >= 0.1 and silent_time < 50.0")
        .silent_time.round(4)
        .value_counts()
        .to_frame()
    )
    silent_time_distribution = df_silent_time.divide(df_silent_time.sum())
    return silent_time_distribution


def get_avalanche_size_distribution(spike_events, silent_threshold=0.2):
    allowed_deltas = np.arange(0, silent_threshold, 0.1)

    df_spike_events = (
        pd.DataFrame(spike_events).sort_values("times").reset_index(drop=True)
    )

    df_spike_events["in_avalanche"] = (
        (df_spike_events["times"].shift(-1) - df_spike_events["times"].shift(0))
        .round(5)
        .isin(allowed_deltas)
    )
    df_spike_events["avalanche_id"] = (
        ~df_spike_events["in_avalanche"]
    ).cumsum() * df_spike_events["in_avalanche"]

    df_avalanche_sizes = (
        df_spike_events.query("avalanche_id != 0")
        .groupby(["avalanche_id"])
        .size()
        .add(1)
        .value_counts()
        .to_frame("avalanche_sizes")
    )

    avalanche_size_distribution = df_avalanche_sizes.divide(df_avalanche_sizes.sum())
    return avalanche_size_distribution


def fit_avalanche_sizes_to_power_distribution(
    spike_events, silent_threshold, plot=False
):
    df_avalanche_sizes = get_avalanche_size_distribution(spike_events, silent_threshold)

    X = df_avalanche_sizes.index
    y = df_avalanche_sizes["avalanche_sizes"]

    popt, pcov = curve_fit(func_powerlaw, X, y)
    perr = np.sqrt(np.diag(pcov))
    popt = np.round(popt, 2)

    if plot:
        plt.plot(X, func_powerlaw(X, *popt))

    print("Function to fit: c * x ** (-m)")
    print(
        f"Curve fit: c={popt[0]} +- {round(perr[0], 3)}, m={popt[1]} +- {round(perr[1], 3)}"
    )
    print(f"Parameter error: {perr}")

    return popt[:2]


def plot_avalanche_sizes_distribution(spike_events):
    df_avalanche_sizes = get_avalanche_size_distribution(spike_events)

    df_avalanche_sizes.reset_index().plot(
        logx=True,
        logy=True,
        kind="scatter",
        x="index",
        y="avalanche_sizes",
        title="Avalanche size distribution, best power law fit",
    )


def compute_gamma(m_length, m_activity):
    return (m_length - 1) / (m_activity - 1)
