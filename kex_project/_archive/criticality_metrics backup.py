import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List


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


def get_silent_time_distribution(df_spike_events: List[pd.DataFrame]):
    parsed_spike_events = []
    for df_spike_event in df_spike_events:
        df_spike_event = df_spike_event.sort_values("times").reset_index(drop=True)
        df_spike_event["times"] = df_spike_event["times"]

        df_silent_time = (
            df_spike_event.assign(
                silent_time=df_spike_event["times"]
                - df_spike_event["times"].shift(periods=1)
            )
            .query("silent_time >= 0.1 and silent_time < 50")
            .silent_time.round(4)
            .value_counts()
            .to_frame()
        )
        parsed_spike_events.append(df_silent_time)
    summed_silent_time_frequencies = (
        pd.concat(parsed_spike_events, axis=1)
        .fillna(0)
        .sum(axis=1)
        .to_frame("silent_time")
    )
    silent_time_distribution = summed_silent_time_frequencies.divide(
        summed_silent_time_frequencies.sum()
    ).sort_index()
    # .sort_values("silent_time", ascending=False)
    return silent_time_distribution


def get_avalanche_size_distribution(
    df_spike_events: List[pd.DataFrame], silent_threshold=0.2
):
    allowed_deltas = np.arange(0, silent_threshold + 0.1, 0.1)

    parsed_spike_events = []
    for df_spike_event in df_spike_events:
        df_spike_event = df_spike_event.sort_values("times").reset_index(drop=True)

        df_spike_event["in_avalanche"] = (
            (df_spike_event["times"].shift(-1) - df_spike_event["times"].shift(0))
            .round(5)
            .isin(allowed_deltas)
        )
        df_spike_event["avalanche_id"] = (
            ~df_spike_event["in_avalanche"]
        ).cumsum() * df_spike_event["in_avalanche"]

        df_avalanche_sizes = (
            df_spike_event.query("avalanche_id != 0")
            .groupby(["avalanche_id"])
            .size()
            .add(1)
            .value_counts()
            .to_frame("avalanche_sizes")
        )
        parsed_spike_events.append(df_avalanche_sizes)

    summed_avalanche_frequencies = (
        pd.concat(parsed_spike_events, axis=1)
        .fillna(0)
        .sum(axis=1)
        .to_frame("avalanche_sizes")
    )

    avalanche_size_distribution = summed_avalanche_frequencies.divide(
        summed_avalanche_frequencies.sum()
    )
    # .sort_values("avalanche_sizes", ascending=False)

    return avalanche_size_distribution


def get_avalanche_duration_distribution(
    df_spike_events: List[pd.DataFrame], silent_threshold=0.2
):
    allowed_deltas = np.arange(0, silent_threshold + 0.1, 0.1)

    parsed_spike_events = []
    for df_spike_event in df_spike_events:
        df_spike_event = df_spike_event.sort_values("times").reset_index(drop=True)

        df_spike_event["in_avalanche"] = (
            (df_spike_event["times"].shift(-1) - df_spike_event["times"].shift(0))
            .round(5)
            .isin(allowed_deltas)
        )
        df_spike_event["avalanche_id"] = (
            ~df_spike_event["in_avalanche"]
        ).cumsum() * df_spike_event["in_avalanche"]

        df_avalanche_durations = (
            df_spike_event.query("avalanche_id != 0")
            .groupby(["avalanche_id"])
            .agg({"times": ["min", "max"]})
            .droplevel(0, axis=1)
            .assign(avalanche_duration=lambda x: x["max"] - x["min"] + 0.1)
        )

        df_avalanche_duration_freq = (
            df_avalanche_durations.round(5)
            .groupby(["avalanche_duration"])
            .size()
            .to_frame("avalanche_duration_freq")
        )

        parsed_spike_events.append(df_avalanche_duration_freq)

    summed_avalanche_duration_freq = (
        pd.concat(parsed_spike_events, axis=1)
        .fillna(0)
        .sum(axis=1)
        .to_frame("avalanche_durations")
    )
    avalanche_duration_distribution = summed_avalanche_duration_freq.divide(
        summed_avalanche_duration_freq.sum()
    )
    # .sort_values("avalanche_durations", ascending=False)

    return avalanche_duration_distribution


def fit_avalanche_sizes_to_power_distribution(
    df_spike_events: List[pd.DataFrame],
    silent_threshold,
    plot=False,
    verbose=True,
    return_error=False,
):
    df_avalanche_sizes = get_avalanche_size_distribution(
        df_spike_events, silent_threshold
    ).pipe(remove_truncated_values)

    X = df_avalanche_sizes.index
    y = df_avalanche_sizes["avalanche_sizes"]

    popt, pcov = curve_fit(func_powerlaw, X, y)
    perr = np.sqrt(np.diag(pcov))
    popt = np.round(popt, 2)

    if plot:
        ax = df_avalanche_sizes.reset_index().plot(
            logx=True,
            logy=True,
            kind="scatter",
            x="index",
            y="avalanche_sizes",
            s=5.0,
        )
        ax.set_ylim(10**-4, 1)
        ax.set_xlabel("Avalanche sizes", fontsize=14)
        ax.set_ylabel("Normalized Frequency", fontsize=14)
        ax.plot(X, func_powerlaw(X, *popt))

    if verbose:
        print("Function to fit: c * x ** (-m)")
        print(
            f"Curve fit: c={popt[0]} +- {round(perr[0], 3)}, m={popt[1]} +- {round(perr[1], 3)}"
        )
        print(f"Parameter error: {perr}")
    if return_error:
        return [(p, err) for p, err in zip(popt, perr)]
    return popt[:2]


def fit_silent_period_to_power_distribution(
    df_spike_events,
    plot=False,
    verbose=True,
    return_error=False,
):
    df_silent_time = get_silent_time_distribution(df_spike_events).pipe(
        remove_truncated_values
    )

    X = df_silent_time.index
    y = df_silent_time["silent_time"]

    popt, pcov = curve_fit(func_powerlaw, X, y)
    perr = np.sqrt(np.diag(pcov))
    popt = np.round(popt, 2)

    if plot:
        ax = df_silent_time.reset_index().plot(
            logx=True,
            logy=True,
            kind="scatter",
            x="index",
            y="silent_time",
            s=4.0,
        )
        ax.set_ylim(10**-5, 1)
        ax.set_xlabel("Silent Periods [ms]", fontsize=14)
        ax.set_ylabel("Normalized Frequency", fontsize=14)
        ax.plot(X, func_powerlaw(X, *popt))

    if verbose:
        print("Function to fit: c * x ** (-m)")

        print(
            f"Curve fit: c={popt[0]} +- {round(perr[0], 3)}, m={popt[1]} +- {round(perr[1], 3)}"
        )
        print(f"Parameter error: {perr}")

    if return_error:
        return [(p, err) for p, err in zip(popt, perr)]
    return popt[:2]


def fit_avalanche_duration_to_power_distribution(
    df_spike_events,
    silent_threshold,
    plot=False,
    verbose=True,
    return_error=False,
):
    df_avalanche_duration = get_avalanche_duration_distribution(
        df_spike_events, silent_threshold
    ).pipe(remove_truncated_values)
    X = df_avalanche_duration.index
    y = df_avalanche_duration["avalanche_durations"]

    popt, pcov = curve_fit(func_powerlaw, X, y)
    perr = np.sqrt(np.diag(pcov))
    popt = np.round(popt, 2)

    if plot:
        ax = df_avalanche_duration.reset_index().plot(
            logx=True,
            logy=True,
            kind="scatter",
            x="avalanche_duration",
            y="avalanche_durations",
            s=4.0,
        )
        ax.set_ylim(10**-4, 1)
        ax.set_xlabel("Avalanche durations [ms]", fontsize=14)
        ax.set_ylabel("Normalized Frequency", fontsize=14)
        ax.plot(X, func_powerlaw(X, *popt))

    if verbose:
        print("Function to fit: c * x ** (-m)")
        print(
            f"Curve fit: c={popt[0]} +- {round(perr[0], 3)}, m={popt[1]} +- {round(perr[1], 3)}"
        )
        print(f"Parameter error: {perr}")

    if return_error:
        return [(p, err) for p, err in zip(popt, perr)]
    return popt[:2]


def plot_avalanche_sizes_distribution(spike_events):
    df_avalanche_sizes = get_avalanche_size_distribution(spike_events)
    df_avalanche_sizes

    ax = df_avalanche_sizes.reset_index().plot(
        logx=True,
        logy=True,
        kind="scatter",
        x="index",
        y="avalanche_sizes",
        title="Avalanche size distribution, best power law fit",
    )
    ax.set_ylim(10**-4, 1)
    return df_avalanche_sizes


def compute_gamma(m_duration, m_size, includes_errors):
    if includes_errors:
        max_gamma = (m_duration[0] + m_duration[1] - 1) / (m_size[0] + m_size[1] - 1)
        min_gamma = (m_duration[0] - m_duration[1] - 1) / (m_size[0] - m_size[1] - 1)

        mean_gamma = np.mean([max_gamma, min_gamma])
        err_gamma = np.abs(max_gamma - min_gamma)

        return (mean_gamma, err_gamma)

    return (m_duration - 1) / (m_size - 1)


def remove_truncated_values(distribution: pd.DataFrame):
    first_occurence_idx = distribution.idxmin()[0]
    return distribution.loc[:first_occurence_idx]
