from packages.utils.plot_utils import plot_gamma_model_fit
import numpy as np
import pickle
from packages.utils.figure_configuration_ieee_standard import figure_configuration_ieee_standard
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

for num in range(4):
    file_name='gamma_service_'+str(num)
    data=np.load(file_name+'.npz',allow_pickle=True)

    with open(file_name+'_trace.pkl', 'rb') as file:
        trace = pickle.load(file)

    figure_configuration_ieee_standard()

    data=data['data']
    x_plot = np.linspace(np.floor(data.min()), np.ceil(data.max()) , 200)
    post_pdfs = gamma.pdf(np.atleast_2d(x_plot), trace['alpha'][:,np.newaxis], scale=
    1 / trace['beta'][:,np.newaxis])
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [5, 95], axis=0)
    fig, ax = plt.subplots()
    ax.hist(data, density=True, color='gray', lw=0, alpha=0.5,bins=100)
    ax.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray')
    # ax.plot(x_plot, post_pdfs[0], c='gray', label='Posterior sample densities')
    # ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0), c='k')
    ax.set_xlabel(r'service time $v$ [s]')
    ax.set_ylabel(r'$f_V(v)$')
    plt.xlim((1,4))
    plt.ylim((0,1))
    # ax.legend(loc=2)


    plt.savefig('gamma_plot_' + str(num) + '.pdf')
    plt.show()
