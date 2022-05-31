import numpy as np
import pymc3 as pm
from scipy.stats import norm, gamma, expon
import matplotlib.pyplot as plt

# TOOD Document
def plot_expon_model_fit(mu, data, savefig=True, figname=None, xlabel='service times'):
    if figname is None:
        figname = 'postMean'

    # Plot
    # pm.plots.traceplot(trace)
    # if savefig:
    #     plt.savefig('trace_plot_' + figname + '.pdf')

    # plt.show()

    x_plot = np.linspace(np.floor(data.min()) - 1, np.ceil(data.max()) + 1, 200)
    post_pdf_contribs = expon.pdf(np.atleast_3d(x_plot), mu)
    post_pdfs = (post_pdf_contribs).sum(axis=-1)

    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(data, density=True, color='blue', lw=0, alpha=0.5, bins=100)

    ax.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray', alpha=0.45)
    # ax.plot(x_plot, post_pdfs[0], c='gray', label='Posterior sample densities')
    # ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0), c='k', label='Posterior expected density')

    ax.set_xlabel(xlabel)
    # ax.set_xlabel('log standardized service times')

    ax.set_ylabel('Density')

    ax.legend(loc=2)
    if savefig:
        plt.savefig('mixture_plot_' + figname + '.pdf')
    plt.show()


# TOOD Document
def plot_mixture_model_fit(trace, data, savefig=True, figname=None, xlabel='service times'):
    if figname is None:
        figname = 'log_mixture_model'

    # Plot
    pm.plots.traceplot(trace)
    if savefig:
        plt.savefig('trace_plot_' + figname + '.pdf')

    plt.show()
    x_plot = np.linspace(np.floor(data.min()) - 1, np.ceil(data.max()) + 1, 200)
    post_pdf_contribs = norm.pdf(np.atleast_3d(x_plot), trace['mu'][:, np.newaxis, :],
                                 trace['tau'][:, np.newaxis, :])
    post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)

    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(data, density=True, color='blue', lw=0, alpha=0.5)

    ax.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray', alpha=0.45)
    ax.plot(x_plot, post_pdfs[0], c='gray', label='Posterior sample densities')
    ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0), c='k', label='Posterior expected density')

    ax.set_xlabel(xlabel)
    # ax.set_xlabel('log standardized service times')

    ax.set_ylabel('Density')

    ax.legend(loc=2)
    if savefig:
        plt.savefig('mixture_plot_' + figname + '.pdf')
    plt.show()


# TOOD Document
def plot_gamma_mixture_model_fit(trace, data, savefig=True, figname=None, xlabel='service times'):
    if figname is None:
        figname = 'gamma_mixture_model'

    # Plot
    pm.plots.traceplot(trace)
    if savefig:
        plt.savefig('trace_plot_' + figname + '.pdf')

    plt.show()

    x_plot = np.linspace(np.floor(data.min()) - .5, np.ceil(data.max()) + .5, 200)
    post_pdf_contribs = gamma.pdf(np.atleast_3d(x_plot), trace['alpha_i'][:, np.newaxis, :], scale=
    1 / trace['beta_i'][:, np.newaxis, :])
    post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=100, density=True, color='blue', lw=0, alpha=0.5)
    ax.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray', alpha=0.45)
    ax.plot(x_plot, post_pdfs[0], c='gray', label='Posterior sample densities')
    ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0), c='k', label='Posterior expected density')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    # ax.legend(loc=2)
    ax.legend(loc='upper right')

    if savefig:
        plt.savefig('mixture_plot_' + figname + '.pdf')
    # plt.show()


# TOOD Document
def plot_gamma_model_fit(trace, data, savefig=True, figname=None, xlabel='service times'):
    if figname is None:
        figname = 'gamma model'

    # Plot
    pm.plots.traceplot(trace)
    if savefig:
        plt.savefig('trace_plot_' + figname + '.pdf')

    plt.show()

    x_plot = np.linspace(np.floor(data.min()) - .5, np.ceil(data.max()) + .5, 200)
    post_pdfs = gamma.pdf(np.atleast_2d(x_plot), trace['alpha'][:,np.newaxis], scale=
    1 / trace['beta'][:,np.newaxis])
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    weights = np.ones_like(data)/float(len(data))
    ax.hist(data, bins=100, density=True, color='blue', lw=0, alpha=0.5, weights=weights)
    ax.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray', alpha=0.45)
    ax.plot(x_plot, post_pdfs[0], c='gray', label='Posterior sample densities')
    ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0), c='k', label='Posterior expected density')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

    if savefig:
        plt.savefig('gamma_plot_' + figname + '.pdf')
    # plt.show()
