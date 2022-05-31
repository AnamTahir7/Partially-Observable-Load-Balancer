from packages.utils.plot_utils import plot_mixture_model_fit
import numpy as np
import pickle

for num in range(1):
    file_name='logMixture_arrival_'+str(num)
    # file_name='logMixture_service_'+str(num)
    data=np.load(file_name+'.npz',allow_pickle=True)

    with open(file_name+'_trace.pkl', 'rb') as file:
        trace = pickle.load(file)
        
    plot_mixture_model_fit(trace, data['log_std_data'], savefig=True, figname=file_name)