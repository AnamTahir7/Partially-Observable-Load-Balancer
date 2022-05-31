import numpy as np
import warnings

warnings.filterwarnings("ignore")


class ModelExtractor:
    """
    Make an instance of the queuing_system model with input parameters
    """

    def __init__(self, arrival_rate: int, queue_lengths: np.ndarray, dept_rates: np.ndarray, N: int, drop_reward: int):
        """
        :param arrival_rate: average rate of arrival
        :param queue_lengths: buffer size of each queue
        :param dept_rates: # average service rates
        :param drop_factor: an additional state for dropping pkts
        :param N:  total number of queues
        :param discount: discount factor
        :param drop_reward: penalty for dropping a pkt
        :param start_state: initial state of the system
        :param delay_rate: delay prob
        """
        self.queue_length = queue_lengths
        self.discount = 0.95
        self.num_queues = N
        self.tot_states = np.tile(self.queue_length + 1, 3)
        self.num_states = np.product(self.tot_states)
        self.num_actions = N  # actions = num of available queues
        self.actions_dict = {}
        for i in range(N):
            self.actions_dict[i] = 'q{}'.format(i)
        self.actions_list = list(self.actions_dict.values())  # including departures
        assert len(self.actions_list) == N
        self.qns = self.num_states
        self.qtile = self.tot_states
        self.curr_state = np.zeros(len(self.qtile), dtype=int)
        self.prev_state = None
        self.dept_rates = dept_rates
        self.drop_reward = drop_reward
        self.arr_rate = arrival_rate
        self.curr_obs = self.curr_state[-N:]  # initially observation = state
        self.prev_obs = self.curr_obs.copy()  # initially observation = state

    def take_action(self, QueuingSimulator, action: str):
        """
        return next state, next obs and reward after taking action on current state
        :param action: current action
        :param curr_obs: current observation
        :return: next state, next observation, reward
        """

        state, observation, reward = QueuingSimulator.simulate(self.curr_state, action)
        self.prev_state = self.curr_state
        return state, observation, reward
