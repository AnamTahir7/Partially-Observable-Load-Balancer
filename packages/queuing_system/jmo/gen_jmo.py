""" Solving the routing problem using jmo-blind algorithm till maximum number of plays """

import numpy as np
from packages.queuing_system.jmo.jmo_env import ModelExtractor
from packages.queuing_system import QueueingSystem


class Jmo(QueueingSystem):
    def __init__(self, max_play: int, arrival_rate: int, shape, queues_lengths: np.ndarray, dept_rates: np.ndarray,
                 N: int, discount: float, drop_reward: int, starting_st: np.ndarray, delay_rate, *args, srv_times,
                   inter_arr_times):
        QueueingSystem.__init__(self, max_play, arrival_rate, shape, queues_lengths, dept_rates, N,
                                discount, drop_reward, starting_st, delay_rate,  ModelExtractor, 'Jmo', srv_times,
                                inter_arr_times)
