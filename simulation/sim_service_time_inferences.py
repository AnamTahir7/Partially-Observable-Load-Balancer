import sys

sys.path.append('..')

import numpy as np
from packages.infer.counting_simulator_inference import InferCountingSimulator
import pickle

if __name__ == '__main__':
    path = '../network data/kaggle/10000/'
    file_name = 'data_st_10000'
    extension = '.npz'
    data_st = np.load(path + file_name + extension)
    service_times = np.array([data_st['service_times']])

    file_name = 'data_iat_10000'
    extension = '.npz'
    data_iat = np.load(path + file_name + extension)
    inter_arrival_time = data_iat['inter_arrival_time']





    nServers = service_times.__len__()
    # inter_arrival_time = 1
    InferEngine = InferCountingSimulator(nComponents=3, save_inferences=True)
    inference_types = ['gammaMixture', 'gamma']
    # inference_types = ['logMixture', 'gammaMixture', 'gamma']
    for inference_type in inference_types:
        print('inference type', inference_type)
        simulator = InferEngine.get_CountingSimulator(inter_arrival_time, service_times, inference_type=inference_type)
        np.savez('pmfs'+inference_type, pmfs=simulator.pmfs)
        with open(inference_type + '_simulator.pkl', 'wb') as file:
            pickle.dump(simulator, file)

    inference_type= 'postMean'
    simulator = InferEngine.get_CountingSimulator(inter_arrival_time, service_times, inference_type=inference_type)
    with open(inference_type + '_simulator.pkl', 'wb') as file:
        pickle.dump(simulator, file)
