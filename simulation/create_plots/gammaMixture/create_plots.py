from packages.utils.plot_utils import plot_gamma_mixture_model_fit
import numpy as np
import pickle

for num in range(4):
    file_name='gammaMixture_service_'+str(num)
    data=np.load(file_name+'.npz',allow_pickle=True)

    with open(file_name+'_trace.pkl', 'rb') as file:
        trace = pickle.load(file)
        
    plot_gamma_mixture_model_fit(trace, data['data'], savefig=True, figname=file_name)