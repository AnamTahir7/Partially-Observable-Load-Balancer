import numpy as np
import random

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
        self.curr_obs = self.curr_state[:N]
        self.d = 2

    def take_action(self, QueuingSimulator, action: int):
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
        choose action from the generated tree that maximises V
        :return: action and its index
        """
        if self.d >= self.num_queues:        #act like JSQ
            curr_state = self.curr_state[:self.num_queues]
            poss_actions = self.actions_list

            if len(set(curr_state)) == 1:
                action = np.random.choice(poss_actions)
                idx = self.actions_list.index(action)

            else:
                curr_min_queues = np.where(curr_state == min(curr_state))[0]
                if len(curr_min_queues) > 1:
                    rand_min_q = np.random.choice(curr_min_queues)
                else:
                    rand_min_q = curr_min_queues[0]
                action = poss_actions[rand_min_q]
                idx = rand_min_q
        else:
            chosen_queues = random.sample(range(self.num_queues), self.d)

            curr_state = self.curr_state[:self.num_queues][chosen_queues]
            poss_actions = [self.actions_list[i] for i in chosen_queues]

            if len(set(curr_state)) == 1:
                action = np.random.choice(poss_actions)
                idx = self.actions_list.index(action)
            else:
                curr_min_queues = np.where(curr_state == min(curr_state))[0]
                if len(curr_min_queues) > 1:
                    rand_min_q = np.random.choice(curr_min_queues)
                else:
                    rand_min_q = curr_min_queues[0]
                action = poss_actions[rand_min_q]
                idx = chosen_queues[rand_min_q]

        return action, idx
