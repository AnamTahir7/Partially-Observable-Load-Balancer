import numpy as np


class ModelExtractor:

    def __init__(self, arrival_rate: int, queue_lengths: np.ndarray, dept_rates: np.ndarray, N: int,
                 discount: float, drop_reward: int):
        """
        :param arrival_rate: average rate of arrival
        :param queue_lengths: buffer size of each queue
        :param dept_rates: # average service rates
        :param drop_factor: an additional state for dropping pkts
        :param N: total number of queues
        :param discount: discount factor
        :param drop_reward: penalty for dropping a pkt
        :param start_state: initial state of the system
        """
        self.queue_length = queue_lengths
        self.discount = discount
        self.num_queues = N
        self.tot_states = np.tile(self.queue_length + 1, 3)
        self.num_states = np.product(self.tot_states)
        self.init_state = None
        self.num_actions = N
        self.actions_dict = {}
        for i in range(N):
            self.actions_dict[i] = 'q{}'.format(i)
        self.actions_list = list(self.actions_dict.values())  # including departures
        assert len(self.actions_list) == N
        self.qtile = self.tot_states
        self.curr_state = np.zeros(len(self.qtile), dtype=int)
        self.prev_state = None
        self.dept_rates = dept_rates
        self.drop_reward = drop_reward
        self.arr_rate = arrival_rate
        self.curr_obs = self.curr_state[-N:] # initially observation = state

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

    def get_action(self):
        """
        80% of the time : choosing the one with max observation, if more than one with max obs then choose one randomly,
        if no obs received then also choose randomly.
        20% of the time: choose a serve randomly no matter what observation is received

        :return: action and its index
        """
        curr_obs = self.curr_state[-self.num_queues:]
        poss_actions = self.actions_list
        eps = np.random.rand(1)

        if eps < 0.8:
            if len(set(curr_obs)) == 1:
                action = np.random.choice(poss_actions)
            else:
                max_obs = np.where(curr_obs == max(curr_obs))[0]
                if len(max_obs) == 1:
                    action = poss_actions[max_obs[0]]
                else:
                    max_queue = np.random.choice(max_obs)
                    action = poss_actions[max_queue]
        else:
            action = np.random.choice(poss_actions)

        idx = self.actions_list.index(action)

        return action, idx
