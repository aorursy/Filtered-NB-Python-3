#!/usr/bin/env python
# coding: utf-8



# define plot functions
import matplotlib.pyplot as plt
from plotly import offline as py
import plotly.figure_factory as ff
import plotly.tools as tls
py.init_notebook_mode()

def box_plot_signal(output_path, scenarios, arr):
    plt.figure(figsize=(10, 6))
    plt.boxplot(arr, labels=scenarios)
    title = output_path.split('/')[-1]
    plt.title(title)
    py.iplot(tls.mpl_to_plotly(plt.gcf()))
    plt.close()


def plot_graphs(output_path, signal, arr):
    binSize = 20
    plt.figure(figsize=(10, 6))
    plt.hist(arr, bins=binSize)
    plt.ylabel('# of times')
    plt.xlabel('range of ' + signal)
    title = output_path.split('/')[-1]
    plt.title(title)
    py.iplot(tls.mpl_to_plotly(plt.gcf()))
    plt.close()


def plot_statistics(output_path, statistics_result, cal_list, scenarios, bioharness_signals):
    for cal in cal_list:
        table = []
        table.append([''] + all_signals)
        for s in scenarios:
            row = []
            row.append(' '.join(s))
            for d in all_signals:
                if cal in statistics_result[s][d].keys():
                    row.append(statistics_result[s][d][cal])
                else:
                    row.append(None)
            table.append(row)

        figure = ff.create_table(table)
        print(cal + ' Statistics')
        py.iplot(figure)




import pickle

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)




import numpy as np

def calculate_statistics_features(arr):
    if not arr:
        return None
    v_mean = np.mean(arr)
    v_mode = max(arr, key = arr.count)
    v_std = np.std(arr)
    v_max = max(arr)
    v_min = min(arr)
    v_median = np.percentile(arr, 50)
    v_Q1 = np.percentile(arr, 25)
    v_Q3 = np.percentile(arr, 75)

    return (v_mean, v_mode, v_median, v_std, v_max, v_min, v_Q1, v_Q3)




"""
This file is used to do the statistics based on specific scenarios, e.g. “local”, “freeFlow”, “local_freeFlow”
"""
__all__ = []

__author__ = 'Ting Zhou <zhouting@umich.edu>'

import os, collections
from scipy.interpolate import interp1d

bioharness_signals = ['HR', 'BR', 'HRV', 'ECGAmplitude']
shimmer_signals = ['id3BE0_GSR_Skin_Resistance_CAL']
scenarios = [('parkingLot', ''), ('local', 'freeFlow'), ('entering', 'freeway'),
             ('exiting', 'freeway'), ('ramp', '')]
cal_list = ['mean', 'mode', 'median', 'std', 'max', 'min', 'Q1', 'Q3']
all_signals = bioharness_signals + shimmer_signals




def find_scenario(input_path, scenario_infras, scenario_traffic=''):
    """find data based on specific scenario

    Args:
        input_path(str): the directory of all the segmented result
        driverID:
        ...

    Yields:
        matched_data (dict): segmented data matches the scenario passed in

    """
    try:
        files = os.listdir(input_path)
    except IOError:
        print("Cannot get the input path directory!")

    matched_data = []

    # Check the input path files' name
    matched_files = []
    for file in files:
        if file.lower().find('segmented_data.pkl') != -1:
            matched_files.append(file)

    for file in matched_files:
        cur_segment = load_obj(input_path + '/' + file)
        for j in range(len(cur_segment)):
            cur_scenario = cur_segment[j]['scenario']

            # parkingLot
            if cur_scenario == 'parkingLot':
                if scenario_infras == 'parkingLot':
                    yield cur_segment[j]
                continue

            # Others
            # except cases like ramp, this does not have '_'
            try:
                cur_scenario_infras, cur_scenario_traffic = cur_scenario.split('_')
            except:
                cur_scenario_infras = cur_scenario
                cur_scenario_traffic = ''

            if cur_scenario_infras == scenario_infras and cur_scenario_traffic == scenario_traffic:
                yield cur_segment[j]


def calculate_based_on_scenario(input_path, scenario_infras, scenario_traffic):
    matched_data = list(
        find_scenario(input_path, scenario_infras, scenario_traffic))

    if not matched_data:
        print('No matched scenario found!!!')
        return None

    result = collections.defaultdict(dict)
    signal_arr = []

    # extract bioharness signals
    for signal in bioharness_signals:
        arr = []
        for data in matched_data:
            arr.extend(data['bio_data'][signal])

        if not arr:
            signal_arr.append(arr)
            print('signal ' + signal + 'not exists')
            continue

        ## temp clean up for HR, BR data, remove 0
        if signal in ['HR', 'BR']:
            arr = [arr[i] for i in range(len(arr)) if arr[i] != 0]

        # arr = list(sig.resample(arr, len(arr) * 10))

        # # Coment out for debug
        # arr = arr[0:10]

        signal_arr.append(arr)

        result[signal] = {}
        cal_result = calculate_statistics_features(arr)

        # set the plot output path
        output_path = input_path + '/graph/' + scenario_infras + '_'                       + scenario_traffic + '_' + signal + '_' + '_histgram.png'
        plot_graphs(output_path, signal, arr)

        for i, cal in enumerate(cal_list):
            result[signal][cal] = cal_result[i]

        print('Complete bioharness signal {}!!!'.format(signal))

    # extract shimmer signals
    for signal in shimmer_signals:
        arr = []
        for data in matched_data:
            arr.extend(data['shimmer_data'][signal])

        if not arr:
            signal_arr.append(arr)
            print('Shimmer signal ' + signal + 'not exists')
            continue

        # arr = sig.resample(arr, len(arr) / 5)

        # # Coment out for debug
        # arr = arr[0:10]

        signal_arr.append(arr)

        result[signal] = {}
        cal_result = calculate_statistics_features(arr)

        # set the plot output path
        output_path = input_path + '/graph/' + scenario_infras + '_'                       + scenario_traffic + '_' + signal + '_' + '_histgram.png'
        plot_graphs(output_path, signal, arr)

        for i, cal in enumerate(cal_list):
            result[signal][cal] = cal_result[i]

        print('Complete shimmer signal {}!!!'.format(signal))

    print(f"{scenario_infras}_{scenario_traffic}: All satatistics complete!!!")
    return [signal_arr, result]


def process_statistics(input_path):
    # if you want to deal with parking lot. Just input the scenario_infras = 'parkingLot'
    # scenario_traffic = ''

    statistics = collections.defaultdict(dict)
    scenario_signal = []  # scenario index : signal arr

    i = 0
    while i < len(scenarios):
        # for i, [scenario_infras, scenario_traffic] in enumerate(scenarios):
        [scenario_infras, scenario_traffic] = scenarios[i]
        print('-----------' + 'Scenario: ' + scenario_infras + ' ' +
              scenario_traffic + ' Start------------')

        # if not such scenario found, it returns None. This handles None.
        return_stat = calculate_based_on_scenario(input_path, scenario_infras, scenario_traffic)
        if not return_stat:
            scenarios.pop(i)
            # scenario_signal.append([[] for _ in range(len(scenarios))])
            print('-----------' + 'Scenario: ' + scenario_infras + ' ' +
                  scenario_traffic + 'End-------------\n')
            continue

        signal_arr, statistics[(scenario_infras, scenario_traffic)] = return_stat

        scenario_signal.append(signal_arr)

        print('-----------' + 'Scenario: ' + scenario_infras + ' ' +
              scenario_traffic + 'End-------------\n')
        i += 1

    # Box plot
    for j, signal in enumerate(all_signals):
        arr = []
        for i, [scenario_infras, scenario_traffic] in enumerate(scenarios):
            arr.append(scenario_signal[i][j])

        boxplot_output_path = input_path + '/boxplot/boxplot_' + signal + '.png'
        box_plot_signal(boxplot_output_path, scenarios, arr)

    # plot statistics
    plot_statistics(input_path + '/cal', statistics, cal_list, scenarios, all_signals)




if __name__ == '__main__':
    input_path = '../input'
    process_statistics(input_path)

