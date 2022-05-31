import numpy as np
from abc import ABC, abstractmethod
from packages.utils.utils import sampleDiscrete
import time

class CountingSimulator(ABC):
    """
    Base class for number of customers leaving the parallel queues
    """

    @abstractmethod
    def __init__(self, nServers):
        self.nServers = nServers

    @abstractmethod
    def draw(self):
        pass


class Generic(CountingSimulator):
    """
    Counting Simulator with pmfs as input for the number of customers leaving each queue
    """

    def __init__(self, pmfs):
        """

        :param pmfs: list of np.arrays: each np.array contains probability values for number of customers leaving, e.g. [.1,.2,.7], starting with p(k=0)
        """
        self.pmfs = pmfs
        nServers = self.pmfs.__len__()
        super().__init__(nServers)

    def draw(self):
        """

        :return:  array of ints - number of customers leaving each queue
        """
        sample = np.zeros(self.nServers, int)
        for i in range(self.nServers):
            sample[i] = sampleDiscrete(self.pmfs[i])
        return sample


class GammaArrivalExpService(CountingSimulator):
    """
    Counting simulator for renewal arrival process with gamma increments and iid Exponential service times
    """

    def __init__(self, alpha, beta, mu):
        """

        :param alpha: scalar >0: shape parameter for the gamma process
        :param beta: scalar >0: rate parameter for the gamma process
        :param mu: array of size nServers >0: rate parameter for the exponential service times
        """
        self.alpha = alpha
        self.beta = beta
        # Vector
        self.mu = np.array(mu)
        nServers = self.mu.__len__()
        super().__init__(nServers)

    def draw(self):
        """

        :return: array of ints - number of customers leaving each queue
        """
        # sample = np.random.geometric(self.beta / (self.mu + self.beta))
        sample = np.random.negative_binomial(self.alpha * np.ones_like(self.mu), 1 - self.mu / (self.mu + self.beta))
        return sample


class DetArrivalExpService(CountingSimulator):
    """
    Counting simulator for deterministic arrival process and renewal process iid Exponential service times
    """

    def __init__(self, inter_arrival_time, mu):
        """

        :param inter_arrival_time: deterministicinter-arrival time
        :param mu:  mu: array of size nServers >0: rate parameter for the exponential service times
        """
        self.inter_arrival_time = inter_arrival_time
        # Vector
        self.mu = np.array(mu)
        nServers = self.mu.__len__()
        super().__init__(nServers)

    def draw(self):
        """

        :return: array of ints - number of customers leaving each queue
        """
        sample = np.random.poisson(self.mu * self.inter_arrival_time)
        return sample


# TODO Document
class GammaArrivalParetoService(CountingSimulator):
    def __init__(self, alpha, beta, a, m):
        self.alpha = alpha
        self.beta = beta
        # Vector
        self.a = np.array(a)
        self.m = np.array(m)
        nServers = self.m.__len__()
        super().__init__(nServers)

    def draw(self):
        # begin = time.time()
        interarrival_time = np.random.gamma(self.alpha, 1 / self.beta)
        sample = np.zeros(self.nServers)
        cum_time = np.zeros(self.nServers)
        not_enough_samples = np.array([True] * self.nServers)
        while not_enough_samples.any():
            service_time_all = np.random.pareto(self.a) * self.m
            cum_time += service_time_all
            sample[np.where(cum_time <= interarrival_time)[0]] += 1
            enough_samples_idx = np.where(cum_time > interarrival_time)[0]
            if len(enough_samples_idx) > 0:
                not_enough_samples[enough_samples_idx] = False
        # print('draw', time.time() - begin)
        return sample


    # def draw(self):
    #     begin = time.time()
    #     interarrival_time = np.random.gamma(self.alpha, 1 / self.beta)
    #     sample = np.zeros(self.nServers)
    #     for server in range(self.nServers):
    #         cum_time = 0
    #         not_enough_samples = True
    #         while not_enough_samples:
    #             service_time = np.random.pareto(self.a[server]) * self.m[server]
    #             cum_time += service_time
    #             if cum_time <= interarrival_time:
    #                 sample[server] += 1
    #             else:
    #                 not_enough_samples = False
    #     print('draw', time.time() - begin)
    #     return sample
