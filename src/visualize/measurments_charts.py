import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import storage.constants as db_constants
from analyze.measurement import constants as measurement_constants
from simulation.config import SimulationType

BASE_RESULT_PATH = "../../resources/uniform_dist_result"


def to_visualize_x_label(to_convert):
    if to_convert == db_constants.MIO_SIGMA:
        return "stubbornness standard deviation"
    elif to_convert == db_constants.SWITCH_AGENT_SIGMA:
        return "turnover divergence standard deviation"
    elif to_convert == db_constants.SWITCH_AGENT_RATE:
        return "turnover mean rate"
    elif to_convert == db_constants.RADICAL_EXPOSURE_ETA:
        return "radical exposure parameter (η)"
    elif to_convert == db_constants.MIO:
        return "μ"
    else:
        return to_convert.capitalize()


def to_visualize_name(to_convert):
    if to_convert == measurement_constants.BINS_VARIANCE:
        return "average bin standard deviation"
    elif to_convert == measurement_constants.SPREAD:
        return "spread"
    elif to_convert == measurement_constants.COVERED_BINS:
        return "covered bins"
    elif to_convert == measurement_constants.DISPERSION:
        return "standard deviation"
    elif to_convert == db_constants.MIO_SIGMA:
        return "stubbornness divergence"
    elif to_convert == SimulationType.REPULSIVE:
        return "Repulsion and Attraction Dynamics"
    elif to_convert == SimulationType.SIMILARITY:
        return "Only Attraction Dynamics"
    elif to_convert == db_constants.SWITCH_AGENT_SIGMA:
        return "turnover divergence"
    elif to_convert == db_constants.SWITCH_AGENT_RATE:
        return "turnover rate"
    elif to_convert == db_constants.RADICAL_EXPOSURE_ETA:
        return "radical exposure"
    else:
        return to_convert.capitalize()


def read_basic_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df[db_constants.X] == 9999]
    return df


def plot_measurement_for_simulation(df, measurement_type, x_column, simulation_type, inner_filters, ax, with_title):
    filtered_df = df[df[db_constants.MEASUREMENT_TYPE] == measurement_type]
    filtered_df = filtered_df[filtered_df[db_constants.SIMULATION_TYPE] == simulation_type.name]
    if inner_filters:
        for inner_filter in inner_filters:
            if inner_filter[1] is None:
                inner_filtered_df = filtered_df[filtered_df[inner_filter[0]].isna()]
            else:
                inner_filtered_df = filtered_df[filtered_df[inner_filter[0]] == inner_filter[1]]
            plot_x_y(inner_filtered_df, x_column, ax, color=inner_filter[2])
    else:
        plot_x_y(filtered_df, x_column, ax)
    ax.set_xlabel(to_visualize_x_label(x_column))
    if with_title:
        ax.set_title(f"{to_visualize_name(with_title).title()}")


def plot_all_charts(df, x_column, mechanism_name, inner_filters=None):
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    # Create 8 charts
    for measurement_type in [measurement_constants.BINS_VARIANCE, measurement_constants.SPREAD,
                             measurement_constants.COVERED_BINS]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plot_measurement_for_simulation(df, measurement_type, x_column, SimulationType.REPULSIVE, inner_filters, ax1,
                                        with_title=SimulationType.REPULSIVE)
        plot_measurement_for_simulation(df, measurement_type, x_column, SimulationType.SIMILARITY, inner_filters, ax2,
                                        with_title=SimulationType.SIMILARITY)
        ax1.set_ylabel(to_visualize_name(measurement_type))
        plt.show()


def plot_x_y(filtered_df, x_column, ax, color=None, label=None):
    filtered_df = filtered_df.drop_duplicates([
        db_constants.MEASUREMENT_TYPE, db_constants.SIMULATION_TYPE, db_constants.NUM_OF_AGENTS,
        db_constants.NUM_ITERATIONS, db_constants.MIO, db_constants.MIO_SIGMA, db_constants.NUM_OF_REPETITIONS,
        db_constants.SWITCH_AGENT_RATE, db_constants.SWITCH_AGENT_SIGMA, db_constants.RADICAL_EXPOSURE_ETA,
        db_constants.EPSILON
    ])
    # Define the x and y values for the line chart
    x = filtered_df[x_column]
    y = filtered_df["value"]
    # Calculate the confidence interval using the sample standard deviation
    sample_std = filtered_df["sample_std"]
    n = len(filtered_df)
    # t-value for 95% confidence interval and n-1 degrees of freedom
    t = 2.093
    error = t * (sample_std / np.sqrt(n))
    upper_bound = y + error
    lower_bound = y - error
    # print x and y
    df_for_table = pd.DataFrame({'x': x, 'y': y, 'upper_bound': upper_bound, 'lower_bound': lower_bound})
    # pretty print df as table
    print(df_for_table.to_string(index=False))

    # Create the line chart with confidence interval)
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.3, color=color)


def stubbornness_by_sigma_mio(csv_path):
    df = read_basic_df(csv_path)
    df = df[df[db_constants.MIO] == 0.20000]
    x_column = db_constants.MIO_SIGMA
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.REPULSIVE, None,
                                    axs[0, 0], with_title=SimulationType.REPULSIVE)
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.SIMILARITY, None,
                                    axs[0, 1], with_title=SimulationType.SIMILARITY)
    axs[0, 0].set_ylabel(to_visualize_name(measurement_constants.SPREAD).title())

    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.REPULSIVE, None,
                                    axs[1, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.SIMILARITY, None,
                                    axs[1, 1], with_title=None)
    axs[1, 0].set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS).title())

    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.REPULSIVE, None,
                                    axs[2, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.SIMILARITY, None,
                                    axs[2, 1], with_title=None)
    axs[2, 0].set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE).title())

    plt.show()


def stubbornness_by_mio(csv_path):
    df = read_basic_df(csv_path)
    x_column = db_constants.MIO
    inner_filters = [(db_constants.MIO_SIGMA, None, None), (db_constants.MIO_SIGMA, 0.075, 'orange')]
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.REPULSIVE, inner_filters,
                                    axs[0, 0], with_title=SimulationType.REPULSIVE)
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.SIMILARITY,
                                    inner_filters, axs[0, 1], with_title=SimulationType.SIMILARITY)
    axs[0, 0].set_ylabel(to_visualize_name(measurement_constants.SPREAD).title())

    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.REPULSIVE,
                                    inner_filters, axs[1, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.SIMILARITY,
                                    inner_filters, axs[1, 1], with_title=None)
    axs[1, 0].set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS).title())

    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.REPULSIVE,
                                    inner_filters, axs[2, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.SIMILARITY,
                                    inner_filters, axs[2, 1], with_title=None)
    axs[2, 0].set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE).title())

    plt.show()


def radical_exposure_by_eta(csv_path):
    df = read_basic_df(csv_path)
    x_column = db_constants.RADICAL_EXPOSURE_ETA
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    # Create 8 charts

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.SIMILARITY, None, ax1,
                                    with_title=None)
    ax1.set_ylabel(to_visualize_name(measurement_constants.SPREAD))
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.REPULSIVE, None, ax1,
                                    with_title=None)
    ax1.set_ylabel(to_visualize_name(measurement_constants.SPREAD))
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.SIMILARITY, None,
                                    ax1, with_title=measurement_constants.BINS_VARIANCE)
    ax1.set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE))
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.SIMILARITY, None,
                                    ax2, with_title=measurement_constants.COVERED_BINS)
    ax2.set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS))
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.REPULSIVE, None,
                                    ax1, with_title=measurement_constants.BINS_VARIANCE)
    ax1.set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE))
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.REPULSIVE, None,
                                    ax2, with_title=measurement_constants.COVERED_BINS)
    ax2.set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS))
    plt.show()


def switch_agent_rate(csv_path):
    df = read_basic_df(csv_path)
    x_column = db_constants.SWITCH_AGENT_RATE
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.REPULSIVE, None,
                                    axs[0, 0], with_title=SimulationType.REPULSIVE)
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.SIMILARITY, None,
                                    axs[0, 1], with_title=SimulationType.SIMILARITY)
    axs[0, 0].set_ylabel(to_visualize_name(measurement_constants.SPREAD).title())

    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.REPULSIVE, None,
                                    axs[1, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.SIMILARITY, None,
                                    axs[1, 1], with_title=None)
    axs[1, 0].set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS).title())

    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.REPULSIVE, None,
                                    axs[2, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.SIMILARITY, None,
                                    axs[2, 1], with_title=None)
    axs[2, 0].set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE).title())

    plt.show()


def switch_agent_sigma(csv_path):
    df = read_basic_df(csv_path)
    x_column = db_constants.SWITCH_AGENT_SIGMA
    df[x_column] = df[x_column].astype(float)
    df = df.sort_values(by=x_column, ignore_index=True)

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.REPULSIVE, None,
                                    axs[0, 0], with_title=SimulationType.REPULSIVE)
    plot_measurement_for_simulation(df, measurement_constants.SPREAD, x_column, SimulationType.SIMILARITY, None,
                                    axs[0, 1], with_title=SimulationType.SIMILARITY)
    axs[0, 0].set_ylabel(to_visualize_name(measurement_constants.SPREAD).title())

    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.REPULSIVE, None,
                                    axs[1, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.COVERED_BINS, x_column, SimulationType.SIMILARITY, None,
                                    axs[1, 1], with_title=None)
    axs[1, 0].set_ylabel(to_visualize_name(measurement_constants.COVERED_BINS).title())

    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.REPULSIVE, None,
                                    axs[2, 0], with_title=None)
    plot_measurement_for_simulation(df, measurement_constants.BINS_VARIANCE, x_column, SimulationType.SIMILARITY, None,
                                    axs[2, 1], with_title=None)
    axs[2, 0].set_ylabel(to_visualize_name(measurement_constants.BINS_VARIANCE).title())

    plt.show()


def main():
    stubbornness_by_sigma_mio(f'{BASE_RESULT_PATH}/stubbornness/combined_measurements.csv')
    stubbornness_by_mio(f'{BASE_RESULT_PATH}/stubbornness/combined_measurements.csv')
    radical_exposure_by_eta(f'{BASE_RESULT_PATH}/radical_exposure/combined_measurements.csv')
    switch_agent_rate(f'{BASE_RESULT_PATH}/switch_agent_rate/combined_measurements.csv')
    switch_agent_sigma(f'{BASE_RESULT_PATH}/switch_agent_sigma/combined_measurements.csv')


if __name__ == '__main__':
    main()
