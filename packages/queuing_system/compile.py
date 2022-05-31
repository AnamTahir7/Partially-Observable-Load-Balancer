"""This file contains all the functions required to retrieve stored data in json format and average them"""
from pathlib import Path
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict
from packages.queuing_system.plotting import QueuingResultPlotting
from tqdm import tqdm
import os


class Compile:
    def __init__(self, path, expected_iter, hold_keys=('reward', 'pkt_drp', 'state_action', 'delay',
                                                       'b1_orig', 'b2_orig',
                                                       ),
                                                    start_index=0, stop_index=None, pomcp_model_comparison=False):

        if type(path) is list:
            _files = []
            for i in range(len(path)):
                _path = Path(path[i])
                _filepaths = _path.glob('**/*.json')
                _files.extend([f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                                 or 'eps' in str(f) or 'comp_only_one_data' in str(f))])
            self._filepaths = _files
        else:
            self.path = Path(path)
            _filepaths = self.path.glob('**/*.json')
            self._filepaths = [f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                             or 'eps' in str(f))]
        self.pomcp_comparison = pomcp_model_comparison
        self.expected_iter = expected_iter
        print("Total files found: {}".format(len(self._filepaths)))
        self.tot_files = len(self._filepaths)
        self.sum_data = defaultdict(lambda: defaultdict(list))
        self.hold_keys = hold_keys
        self.hold_data = defaultdict(lambda: defaultdict(list))
        self.start_index = start_index
        self.stop_index = stop_index
        self.model_count = {}
        self.pkt_drp_epoch = {}

    @staticmethod
    def load(path: Path):
        with path.open("r") as jf:
            return json.load(jf)

    def reset(self):
        self.sum_data = defaultdict(lambda: defaultdict(list))
        self.hold_data = defaultdict(lambda: defaultdict(list))

    def compile(self):
        for path in tqdm(self._filepaths, desc='Compiling'):
            _data = self.load(path)
            if self.pomcp_comparison:
                _data = {path.parent.name: _data['pos']}

            converted_data = self._convert_to_numpy(_data)
            self.hold(deepcopy(converted_data), self.hold_keys)
            self.sum(converted_data)
        return self

    def hold(self, data: dict, keys: list):
        """ Stores the data for the given keys """
        data = self.slice_data(data, self.start_index, self.stop_index)

        for model_name, model_data in data.items():
            if not isinstance(model_data, dict):
                continue

            for key in keys:
                if key in model_data:
                    self.hold_data[model_name][key].append(model_data[key])

    @staticmethod
    def slice_data(data: dict, start: int = 0, stop:int=None, ignore_keys=('tot_arr_q',
                                                                           'state_action',
                                                                           'b1_orig', 'b2_orig'
                                                                           )):
        """ Slices data starting from specified start index """
        for model_name, model_data in data.items():
            if not isinstance(model_data, dict):
                continue

            for key, value in model_data.items():
                if any(key.startswith(ik) for ik in ignore_keys):
                    continue

                if isinstance(value, dict):
                    for key1, value1 in value.items():
                        # print(model_name, key1)
                        data[model_name][key][key1] = value1[start:stop]
                else:
                    data[model_name][key] = value[start:stop]
        return data

    def sum(self, data: dict):
        """ Updates self.data with the its sum with data """
        data = self.slice_data(data, self.start_index, self.stop_index)

        if not len(self.sum_data):
            self.sum_data.update(data)
            for model_name in self.sum_data.keys():
                self.model_count[model_name] = 1
            return self.sum_data

        for model_name, model_data in data.items():
            if not isinstance(model_data, dict):
                continue

            if model_name not in self.sum_data:
                self.sum_data[model_name].update(model_data)
                self.model_count[model_name] = 1
                continue

            for key, value in model_data.items():
                if isinstance(value, dict):
                    for key1, value1 in value.items():
                        self.sum_data[model_name][key][key1] += value1
                elif key not in ['q_orig']:
                    self.sum_data[model_name][key] += value
            self.model_count[model_name] += 1

        return self.sum_data

    def _convert_to_numpy(self, data: dict):
        """ Convert data to numpy for easy manipulation """
        _dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                _dict[key] = self._convert_to_numpy(value)
            elif isinstance(value, list):
                _dict[key] = np.array(value)
            else:
                _dict[key] = value
        return _dict

    def average(self):
        """ Takes the average of self.data with total filepaths """
        ret_data = {}
        for model_name, model_data in self.sum_data.items():
            if not isinstance(model_data, dict):
                ret_data[model_name] = model_data
                continue

            # model_values = model_data.get('values')
            ret_data[model_name] = {}

            for key, value in model_data.items():
                if isinstance(value, dict):
                    ret_data[model_name][key] = {}
                    for key1, value1 in value.items():
                        ret_data[model_name][key][key1] = value1 / self.model_count[model_name]#len(self._filepaths)
                else:
                    if key == 'delay':
                        ret_data[model_name][key] = np.divide(value, (self.model_count[model_name] - model_data['pkt_drp'])) # len(self._filepaths)
                    else:
                        ret_data[model_name][key] = np.divide(value, self.model_count[model_name]) # len(self._filepaths)

        return ret_data

    @staticmethod
    def nonzero_average(data: list):
        np_data = np.array(data)
        nz_avg = np.sum(np_data, axis=0) / np.count_nonzero(np_data, axis=0)
        nz_avg = np.nan_to_num(nz_avg, copy=False)
        return nz_avg

    def stats(self):
        """ Returns mean of all data, variance and cumulative sum of hold keys """
        data = self.average()
        # Cumulative sum
        for model_name, model_data in data.items():
            if not isinstance(model_data, dict):
                continue

            for key in self.hold_keys:
                if key in model_data and key == 'reward':
                    data[model_name][key + '_cumsum'] = np.cumsum(model_data[key])

        # Variance
        for model_name, model_data in self.hold_data.items():
            for hkey, hdata in model_data.items():
                if hkey == 'reward':
                    data[model_name][hkey + '_std'] = np.std(hdata, axis=0)
                elif hkey == 'pkt_drp':
                    data[model_name][hkey + '_flatsum'] = np.sum(hdata, axis=1)
                    data[model_name][hkey + '_rate'] = np.sum(hdata, axis=1)/len(hdata[0])

                    data[model_name][hkey + 'drop'] = np.zeros(5)

                    for j in range(len(hdata)):
                        i = 0
                        curr_data = hdata[j]
                        for l in range(5):
                            data[model_name][hkey + 'drop'][l] += curr_data[i:i + 100].sum()
                            i += 100

                    data[model_name][hkey + 'drop'] = data[model_name][hkey + 'drop'] / len(hdata)
                    data[model_name][hkey + 'drop_cumsum'] = np.cumsum(data[model_name][hkey + 'drop'])

                elif hkey == 'action_list':
                    data[model_name][hkey + 'sum'] = np.zeros(5)
                    for j in range(len(hdata)):
                        i = 0
                        curr_data = hdata[j]
                        for l in range(5):
                            data[model_name][hkey + 'sum'][l] += curr_data[i:i + 100].sum()
                            i += 100
                    data[model_name][hkey + 'sum'] = data[model_name][hkey + 'sum'] / len(hdata)
                    data[model_name][hkey + 'cumsum'] = np.cumsum(data[model_name][hkey + 'sum'])

                    # data['pkt_drp_norm'][model_name] = data[model_name][hkey + '_rate']
                elif hkey == 'state_action':
                    data[model_name][hkey + '_nzavg'] = hdata
                elif hkey == 'belief_state_action':
                    data[model_name][hkey + '_nzavg'] = hdata
                elif hkey == 'delay':
                    data[model_name][hkey + '_all'] = hdata
                    temp = np.ravel(hdata)
                    data[model_name][hkey + '_org'] = temp[temp > 0]
                elif hkey == 'b1_orig':
                    data[model_name][hkey + '_all'] = hdata
                elif hkey == 'b2_orig':
                    data[model_name][hkey + '_all'] = hdata
                elif hkey == 'tot_arr_q':
                    for i in range(len(hdata[0])):
                        data[model_name][hkey + '{}'.format(i)] = [item[i] for item in hdata]
        return data