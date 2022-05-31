import numpy as np
import matplotlib.pyplot as plt
import inspect
from pathos.multiprocessing import ProcessingPool
from numpy_groupies import aggregate
from itertools import repeat, count
from collections import namedtuple

import matplotlib


class configurator:
    @classmethod
    def configurator(cls):
        """Returns a namedtuple named after the class whose fields correspond to the constructor's parameter names and
        contain their default values"""
        params = inspect.signature(cls).parameters
        x = {}
        for key in params.keys():
            if key == 'kwargs':
                continue
            x[key] = params[key].default
        c = namedtuple(cls.__name__, list(x.keys()))
        return c(**x)


def randargmax(b, axis=0):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max(axis=axis, keepdims=True)), axis=axis)


def det2stoch(x, nCats):
    """
    Converts a collection of deterministic category assignments into a stochastic representation with all mass placed at
    the indicated categories.

    :param x: 1d array of integers
    :param nCats: integer indicating total number of categories available (must be greater than maximum value in x)
    :return: [L x nCats] array providing the stochastic representation, where L is the length of x
    """
    l = len(x)
    return aggregate(np.vstack((range(l), x)), 1, size=(l, nCats))


def softmax(x, axis=None):
    """
    Numerically stable implementation of the softmax function.

    :param x: arbitrary array of numbers
    :param axis: axis along which the softmax operation should be performed. default: entire array
    :return: array of same shape as x containing the softmax values
    """
    y = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return y / y.sum(axis=axis, keepdims=True)


def sampleDiscrete(weights, size=1, axis=0, keepdims=False, binsearch=True):
    """
    Generates samples from a set of discrete distributions.

    :param weights: Array of positive numbers representing the (unnormalized) weights of the distributions
    :param size: Integer indicating the number of samples to be generated per distribution
    :param axis: Axis along which the distributions are oriented in the array
    :param binsearch: If true, the distributions are processed sequentially but for each distribution the samples are
        drawn in parallel via binary search (fast for many categories and large numbers of samples). Otherwise, the
        distributions are processed in parallel but samples are drawn sequentially (fast for large number of
        distributions).
    :return: Array containing the samples. The shape coincides with that of the weight array, except that the length of
        the specified axis is now given by the size parameter.
    """
    # cast to numpy array and assert non-negativity
    weights = np.array(weights, dtype=float)
    try:
        assert np.all(weights >= 0)
    except AssertionError:
        raise ValueError('negative probability weights')

    # always orient distributions along the last axis
    weights = np.swapaxes(weights, -1, axis)

    # normalize weights and compute cumulative sum
    weights /= np.sum(weights, axis=-1, keepdims=True)
    csum = np.cumsum(weights, axis=-1)

    # get shape of output array and sample uniform numbers
    shape = (*weights.shape[0:-1], size)
    x = np.zeros(shape, dtype=int)
    p = np.random.random(shape)

    # generate samples
    if binsearch:
        # total number of distributions
        nDists = int(np.prod(weights.shape[0:-1]))

        # orient all distributions along a single axis --> shape: (nDists, size)
        csum = csum.reshape(nDists, -1)
        x = x.reshape(nDists, -1)
        p = p.reshape(nDists, -1)

        # generate all samples per distribution in parallel, one distribution after another
        for ind in range(nDists):
            x[ind, :] = np.searchsorted(csum[ind, :], p[ind, :])

        # undo reshaping
        x = x.reshape(shape)
    else:
        # generate samples in parallel for all distributions, sample after sample
        for s in range(size):
            x[..., s] = np.argmax(p[..., s] <= csum, axis=-1)

    # undo axis swapping
    x = np.swapaxes(x, -1, axis)

    # remove unnecessary axis
    if size == 1 and not keepdims:
        x = np.squeeze(x, axis=axis)
        if x.size == 1:
            x = int(x)

    return x


def bc2xy(p, corners):
    """
    Converts barycentric coordinates on a two-dimensional simplex to xy-Cartesian coordinates.

    Parameters
    ----------
    p: [N x 3] array representing points on the two-dimensional probability simplex
    corners: [3 x 2] array whose rows specify the three corners of the simplex

    Returns
    -------
    [N x 2] array containing the xy-Cartesian coordinates of the points
    """
    assert p.shape[1] == 3
    assert corners.shape == (3, 2)
    return p @ corners


def xy2bc(xy, corners):
    """
    Converts xy-Cartesian coordinates to barycentric coordinates on a two-dimensional simplex.

    Parameters
    ----------
    xy: [N x 2] array containing the xy-Cartesian coordinates of the points
    corners: [3 x 2] array whose rows specify the three corners of the simplex

    Returns
    -------
    [N x 3] array representing points on the two-dimensional probability simplex
    """
    A = np.c_[corners, np.ones(3)]
    b = np.c_[xy, np.ones(xy.shape[0])]
    p = np.clip(np.linalg.solve(A.T, b.T).T, 0, 1)
    return p


def parallel(fun, params, MC=None, passID=True):
    """
    Helper for parallel function evaluation.

    :param fun: Function to be evaluated
    :param params: If MC is provided, params is treated as a single parameter set used for each repetition of the
        function evaluation. Otherwise, params is treated as an iterable providing different parameter sets for each
        function evaluation.
    :param MC: Integer number for repeated function evaluation with a single parameter set (for Monte Carlo simulation).
    :param passID: pass the ID of each thread as additional keyword argument to the function (e.g. for random seed)
    :return: List of return values of all function evaluation.
    """
    # create multiprocessing pool
    pool = ProcessingPool()

    # create iterator and corresponding callable for parallel function evaluations
    if passID:
        f = lambda x: fun(*x[0], id=x[1])
        if MC is None:
            it = ((p, ID) for (p, ID) in zip(params, count()))
        else:
            it = ((p, ID) for (p, ID) in zip(repeat(params, MC), range(MC)))
    else:
        f = lambda x: fun(*x)
        if MC is None:
            it = params
        else:
            it = repeat(params, MC)

    # run threads and return results
    return pool.map(f, it)


_SQRT2 = np.sqrt(2)
def hellinger(p, q):
    """
    Hellinger distance between two distributions
    Parameters
    ----------
    p - first discrete distribution
    q - second discrete distribution

    Returns
    -------
    scalar hellinger distance measure

    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def makePSD(A):
    A_sym = (A + A.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(A)))
    e = np.max([0, -min_eig + 1e-4])
    A = A_sym + e * np.eye(A.shape[0])
    return A


def normalize01(x):
    shifted = x - x.min()
    m = shifted.max()
    if m == 0:
        y = np.ones_like(x)
    else:
        y = x / m
    return y


def distmat2covmat(distmat, scale=0.1):
    distmat = 0.5 * (distmat + distmat.T)
    distmat = normalize01(distmat)
    covmat = np.exp(-0.5*((distmat/scale)**2))
    covmat = makePSD(covmat)
    return covmat


def positions2distmat(positions):
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    return dists


def MCplot(y, x=None, labels=None, cmap=plt.get_cmap('tab10')):
    """
    Create a shaded mean-std plot for multiple data sets.

    :param y: Input data. Either an [D x MC x N] numpy array or list of length D containing [MC(d) x N(d)] numpy
    arrays,  where D is the number of data sets, MC is the number of Monte Carlo samples, and N is the number of values
        of the free iteration parameter.
    :param x: [N] array containing the values of the free parameter. If None, x is set to [1, ..., N(d)].
    :param labels: iterable containing D labels for the plots
    :param cmap: colormap for the plots
    """
    # iterate over all data sets
    for i, vals in enumerate(y):
        # default x-axis
        if x is None:
            x = range(vals.shape[1])

        # compute mean and std
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)

        # create plots
        plt.plot(x, mean, color=cmap.colors[i])
        plt.fill_between(x, mean-std, mean+std, alpha=0.2, color=cmap.colors[i])

    # add legend
    if labels:
        plt.legend(labels)


def figure_configuration_ieee_standard():
    # IEEE Standard Figure Configuration - Version 1.0

    # run this code before the plot command

    #
    # According to the standard of IEEE Transactions and Journals:

    # Times New Roman is the suggested font in labels.

    # For a singlepart figure, labels should be in 8 to 10 points,
    # whereas for a multipart figure, labels should be in 8 points.

    # Width: column width: 8.8 cm; page width: 18.1 cm.

    # width & height of the figure
    k_scaling = 1

    k_width_height = 1.5#1.3  # width:height ratio of the figure

    # fig_width = 17.6/2.54 * k_scaling
    fig_width = 8.8/2.54 * k_scaling
    fig_height = fig_width / k_width_height


    params = {'axes.labelsize': 12,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 12,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': ['Times New Roman'],
              'lines.linewidth': 2.5,
              'axes.linewidth': 1,
              'axes.grid': True,
              'savefig.format': 'pdf',
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'savefig.pad_inches': 0.04,
              'legend.markerscale': 0.9,
              'savefig.bbox': 'tight',
              'lines.markersize': 2,
              'legend.numpoints': 4,
              'legend.handlelength': 2.0, #was 3.5
              'text.usetex': True
              }

    matplotlib.rcParams.update(params)
