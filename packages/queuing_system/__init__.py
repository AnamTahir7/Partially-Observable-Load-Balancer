""" Super class for creating a System """

import numpy as np
from packages.queuing_system.model import SystemModel
from tqdm import tqdm


class QueueingSystem:
    def __init__(self, max_play: int, arrival_rate: int, shape:int, queues_lengths: np.ndarray, dept_rates: np.ndarray,
                 N: int, discount: float, drop_reward: int, starting_st: np.ndarray, delay_rate: np.ndarray,
                 model_extractor, algorithm: str, srv_times: dict, inter_arr_times:np.ndarray):

        """
        :param max_play: total number of arrivals and departures
        :param arrival_rate: average rate with which an arrival occurs
        :param queues_lengths: buffer size of each queue
        :param dept_rates: service rate of each queue
        :param N: Total number of queues
        :param discount: discount factor
        :param drop_reward: penalty for dropping a packet
        :param starting_st: initial state of the system
        :param model_extractor: reference to the model extractor class
        """

        self.max_play = max_play
        self.arrival_rate = arrival_rate
        self.queues_lengths = queues_lengths
        self.dept_rates = dept_rates
        self.N = N
        self.discount = discount
        self.drop_reward = drop_reward
        self.starting_st = starting_st
        self.delay_rate = delay_rate
        self.state_list = []
        self.action_list = []
        self.belief_state_list = []

        # simulator
        self.tot_time = [0]
        self.cumtime = [0]
        self.tot_time_passed = [0]
        self.inter_arr_times = inter_arr_times
        self.arr_counter = 0

        self.srv_times = srv_times
        self.wait_times = {}
        self.last_arr_time = {}
        self.arrQ = {}
        self.response_time_all = {}
        self.response_time_remaining = {}
        self.dept_counter = np.zeros(N, int)
        self.all_pkts_in_air = np.zeros(N, int)
        self.s = np.ones(5) * 2
        self.m = 1/dept_rates

        for a in range(N):
            self.wait_times['q{}'.format(a)] = [0]
            self.response_time_all['q{}'.format(a)] = []
            self.last_arr_time['q{}'.format(a)] = [0]
            self.arrQ['q{}'.format(a)] = 0
            self.response_time_remaining['q{}'.format(a)] = []

        self.job_ident = ()
        # Model Extractor
        self.model_extractor = model_extractor

        self.algorithm = algorithm

    @classmethod
    def from_config(cls, p, j, srv_times, inter_arr_times):
        return cls(p.get('max_play_per_mc'), p.get('arrival_rate')[j], p.get('arrival_rate_alpha')[j], p.get('queues_lengths'),
                   p.get('dept_rates'), p.get('N'), p.get('discount'),
                   p.get('drop_reward'), p.get('starting_state'), p.get('delay_prob'), None, '', srv_times=srv_times,
                   inter_arr_times=inter_arr_times)

    def histogram(self, model_values):
        bins = []
        for i in range(self.N):
            model_values.pmf_q_orig[i + 1], bins = np.histogram(model_values.q_orig[i + 1],
                                                                bins=range(0, self.queues_lengths[i] + 2), density=True)
            model_values.pmf_q_obs[i + 1], _ = np.histogram(model_values.q_obs[i + 1],
                                                            bins=range(0, self.queues_lengths[i] + 2), density=True)
        model_values.pmf_bins = bins
        return model_values

    def run(self, model_values: SystemModel, queuing_simulator) -> SystemModel:

        model = self.model_extractor(self.arrival_rate, self.queues_lengths, self.dept_rates,
                                     self.N, self.discount, self.drop_reward)
        _curr_s = model.curr_state.copy()
        _curr_obs = _curr_s[-self.N:]

        if model.num_queues == 2:
            for i in range(model.num_queues):
                model_values.q_orig[i + 1].append(int(_curr_s[i]))
                model_values.q_obs[i + 1].append(int(_curr_obs[i]))

        for i in tqdm(range(self.max_play), desc=self.algorithm, ncols=40):
            action_jsq, action_idx = model.get_action()
            for j in range(model.num_queues):
                if j == action_idx:
                    model_values.tot_arr_q[j] += 1

            self.delay_sim(model_values, action_idx, i, model, queuing_simulator)
            if model.num_queues == 2:
                for j in range(model.num_queues):
                    model_values.q_orig[j + 1].append(int(model.curr_state[j]))
                    model_values.q_obs[j + 1].append(int(model.curr_obs[j]))

        model_values.action_list = self.action_list
        return model_values

    def delay_sim(self, model_values, action, no, model, queuing_simulator):
        # using lindley equation
        curr_s = model.curr_state.copy()
        N = model.num_queues
        ea = np.zeros(N)
        self.dept_counter = np.zeros(N, int)
        currQ = action
        _curr_s = np.array(curr_s[0:N], int)
        _curr_belief = curr_s[N:2*N]
        self.state_list.append(_curr_s.copy())
        self.action_list.append(int(currQ))
        model.prev_obs = model.curr_obs.copy()

        currtau = self.inter_arr_times[no]
        self.tot_time.append(currtau)           #time of each arrival
        self.tot_time_passed = np.cumsum(self.tot_time)     # actual time at that arrival

        #calculate possible departures from previous jobs in this epoch: if departure time of that job is greater than the current time
        for i in range(model.num_queues):
            if curr_s[i] > 0:       #dept possible
                # get all in that queue right now
                jobs = [jobs for jobs in self.job_ident if jobs[1] == i]
                if len(jobs) > 0:
                    ini_jobs = len(jobs)
                    jobs = [i for i in jobs if i[5] > self.tot_time_passed[-1]]
                    curr_jobs = len(jobs)
                    tot_dep = ini_jobs - curr_jobs
                    self.dept_counter[i] += tot_dep

        # get how many ack reach back in this time
        k = self.dept_counter
        ea[currQ] = 1
        curr_s[0:N] = np.maximum((curr_s[0:N] - k), np.zeros(N))

        if curr_s[0:N][currQ] < model.queue_length[currQ]:
            curr_s[0:N] = curr_s[0:N] + ea
            self.cumtime= np.cumsum(self.tot_time)

            self.last_arr_time['q{}'.format(currQ)].append(self.cumtime[-1])
            tau = self.last_arr_time['q{}'.format(currQ)][-1] - self.last_arr_time['q{}'.format(currQ)][-2] # tau for each particular queue

            wt = max(((self.wait_times['q{}'.format(currQ)][self.arrQ['q{}'.format(currQ)]] +
                      self.srv_times['q{}'.format(currQ)][self.arrQ['q{}'.format(currQ)]]) - tau), 0)

            self.arrQ['q{}'.format(currQ)] += 1
            d = wt + self.srv_times['q{}'.format(currQ)][self.arrQ['q{}'.format(currQ)]]
            self.wait_times['q{}'.format(currQ)].append(wt)

            # overall job no, queue no, no in queue, response time, arrival time, dept time
            self.job_ident = [k for k in self.job_ident if k[5] > self.tot_time_passed[-1]]
            self.job_ident += ((no, currQ, curr_s[currQ], d, self.cumtime[-1], d + self.cumtime[-1]),)
            model_values.pkt_drp.append(0)
            model_values.delay.append(d)
        else:
            model_values.pkt_drp.append(1)
            model_values.delay.append(0)

        newPackets = np.minimum(model.curr_state[0:N], k) + model.curr_state[N: 2*N]
        self.all_pkts_in_air = newPackets
        l = np.random.binomial(newPackets, self.delay_rate)

        curr_s[N:2*N] = np.minimum((newPackets - l), (model.qtile[N:2*N]-1))
        curr_s[2*N:] = np.minimum(l, (model.qtile[2*N:]-1))

        model.curr_state = curr_s.copy()
        model.curr_obs = l.copy()
        model_values.reward.append(queuing_simulator.RewardQueuingSimulatorInstance.get_reward(curr_s[0:N], l, currQ))




