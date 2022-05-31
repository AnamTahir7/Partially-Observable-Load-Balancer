import time
from collections import Counter
from typing import Union

import numpy as np

from packages.queuing_system.pos.belief_tree import BeliefTree
import scipy.stats as st

class SolverCreator:

    def __init__(self, model, C, no_particles, inter_arr_times):
        """
        solving the routing problem using queuing_system
        :param model: instance of queuing_system model
        """
        self.model = model
        self.N = self.model.num_queues
        self.C = C
        self.simulation_time = 0.66 * np.mean(inter_arr_times) * np.ones(len(inter_arr_times))  # total time for running the solver before deciding on the action
        self.max_particles = no_particles  # total number of particles for each belief state
        self.reinvigorated_particles_ratio = 0.1  # to avoid particle deprivation
        self.root_particles = []
        for i in range(no_particles):
            self.root_particles.append(self.model.curr_state)
        self.tree = BeliefTree(self.root_particles, model.curr_obs)  # generate a tree with root having these particles in belief space
        self.sampled_state = None
        self.model.ini_belief = None
        self.unack_jobs_each_queue = np.zeros(self.N, dtype=int)
        self.belief_time = 0.34 * np.mean(inter_arr_times)* np.ones(len(inter_arr_times))


    def compute_belief(self):
        """
        computing new belief distribution based on particles added while simulating the tree
        :return: new belief probability distribution
        """
        base = np.zeros(self.model.qns)
        particle_dist = self.particle_distribution(self.tree.root.B)
        for state, prob in particle_dist.items():
            base[state] = round(prob, 6)
        return base

    def rollout_action(self, state: np.ndarray):
        """
        epsilon greedy method
        :param state: current state
        :return: choose an action to take in this state
        """
        poss_actions = self.model.actions_list
        act = np.random.choice(poss_actions)
        a = self.model.actions_list.index(act)
        return a

    def rollout(self, QueuingSimulator, state, h: list, depth: int, max_depth: int):
        """
        random action selection
        Perform randomized recursive rollout search starting from 'h' until the max depth has been achieved
        :param state: starting state
        :param h: history sequence
        :param depth: current planning horizon
        :param max_depth: max planning horizon
        """
        if depth > max_depth:
            return 0
        action = self.rollout_action(state)
        sj, oj, r = QueuingSimulator.simulate(state, action)

        return r + self.model.discount * self.rollout(QueuingSimulator, sj, h + [action, oj],  depth + 1, max_depth)

    def simulator(self, QueuingSimulator, state: np.ndarray, max_depth: int, depth: int = 0, h: list = None,
                  parent=None):
        """
        perform MCTS simulation on POMDP belief search tree stop once max depth reached
        :param state: current sampled state
        :param max_depth: maximum depth for tree
        :param depth: current depth
        :param h: histroy of actions and observations
        :param parent: root node
        :return: Reward for this state
        """

        if depth > max_depth:
            return 0

        obs_h = None if not h else h[-1]
        node_h = self.tree.find_or_create(h, name=str(obs_h) or 'root', parent=parent, observation=obs_h)

        # ROLLOUT
        # Initialize child nodes if not available
        if not node_h.children:
            for i in self.model.actions_list:
                self.tree.add(h + [i], name=i, parent=node_h, action=i)
            return self.rollout(QueuingSimulator, state, h, depth, max_depth)

        # SELECTION - Find the action that maximises the value
        np.random.shuffle(node_h.children)
        max_action = self.ucb1(node_h)
        node_ha = node_h.action_map[max_action]
        # SIMULATION - Perform monte-carlo simulation of the state under the action
        action = self.model.actions_list.index(max_action)
        sj, oj, reward = QueuingSimulator.simulate(state, action)
        R = reward + (self.model.discount * self.simulator(QueuingSimulator, sj, max_depth, depth + 1,
                                                           h=h + [node_ha.action, oj], parent=node_ha))

        # BACK-PROPAGATION - Update the belief node for h
        state = [self.model.queue_length[0] if x > self.model.queue_length[0] else x for x in state]
        node_h.B.append(state)

        node_h.N += 1
        node_h.V += R

        # Update action node for this action
        node_ha.N += 1
        node_ha.V += ((R - node_ha.V) / node_ha.N)

        return R

    def solve(self, T: int, QueuingSimulator, i):
        # solve till depth T
        begin = time.time()
        while time.time() - begin < self.simulation_time[i]:
            state, idx = self.tree.root.sample_state(self.model)
            self.sampled_state = idx
            self.simulator(QueuingSimulator, state, max_depth=T, h=self.tree.root.h)

    def get_action(self, counter):
        """
        choose action from the generated tree that maximises V
        :return: action and its index
        """

        actions = set(self.tree.root.action_map.keys()).intersection(self.model.actions_list)
        action_vals = [(child.V, child.action) for child in self.tree.root.action_map.values()
                       if child.action in actions]

        vals = [child.V for child in self.tree.root.action_map.values()]
        vals_nonzero_idx = np.nonzero(np.array(vals))[0]
        vals_nonzero = np.array(vals)[vals_nonzero_idx]
        prob_ratio = self.model.dept_rates/sum(self.model.dept_rates)
        if len(set(vals)) == 1:     #give high prioirty to faster queues
            child = [child.action for child in self.tree.root.action_map.values()]
            action = np.random.choice(child, p=prob_ratio)
            idx = self.model.actions_list.index(action)
        else:              # send to the one which has given back min acks - to try to explore more
            max_val_idx = vals_nonzero_idx[vals_nonzero.argsort()[-2:][::-1]]
            idx = max_val_idx[np.where(self.unack_jobs_each_queue[max_val_idx] ==
                                                   min(self.unack_jobs_each_queue[max_val_idx]))[0]][0]

        action = action_vals[idx][1]
        return action, idx

    def convert_str_to_array(txt: str):
        """ Converts [0 0] to array type """
        txt = txt.lstrip('[').rstrip(']')
        txt = list(map(int, txt.split(' ')))
        return np.array(txt)

    def sums(self, length, total_sum):
        if length == 1:
            yield (total_sum,)
        else:
            for value in range(total_sum + 1):
                for permutation in self.sums(length - 1, total_sum - value):
                    yield (value,) + permutation

    def update_belief(self, QueuingSimulator, action_str: str, obs: np.ndarray, delay_rate, k_samples, job_no):

        """
        Update the belief tree given the new observation, extending the history, updating particle sets, etc
        Unweighted particle filtering
        :param belief_prob: current belief probalities
        :param action: action taken
        :param obs: current observation
        :return: updated belief probabilities
        """


        root = self.tree.root  # current root
        # find new root based on action taken
        new_root = root.get_child(action_str).get_child(str(obs))
        action = self.model.actions_list.index(action_str)
        self.unack_jobs_each_queue[action] += 1
        self.unack_jobs_each_queue -= obs
        sim_time = self.belief_time[job_no]    # limit the time taken for particle adding
        N = self.model.num_queues
        prev_obs = self.model.prev_obs
        particles = []
        if new_root is None:  # if no new root available create a node in tree
            action_node = root.get_child('q{}'.format(action))

            if action_node:
                new_root = self.tree.add(h=action_node.h + [obs],
                                         name=str(obs) or 'root',
                                         parent=None,
                                         observation=obs,
                                         particle=[])
            else:
                new_root = self.tree.add(h=[obs],
                                         name=str(obs) or 'root',
                                         parent=None,
                                         observation=obs,
                                         particle=[])
        list_sj = []
        list_w = []
        # timeout = 0
        begin = time.time()
        while (time.time() - begin < sim_time):
            si, si_idx = root.sample_state(self.model)
            if np.all(si[-N:] == prev_obs):
                sj, oj, r = QueuingSimulator.simulate(si, action)
                # Assign weight here based on binomial distribution for k o s
                # importance sampling
                sm = sj[0:N] + sj[N:2*N]
                if np.all(sj <= self.model.queue_length[0]) and np.all(oj == obs) and np.all(sm ==
                                                                                             self.unack_jobs_each_queue):
                    if not np.all(list_sj == sj):
                        wi_sum = np.array(np.zeros(N), dtype=float)
                        for m in range(len(k_samples)):
                            ki = np.minimum(k_samples[m], si[0:N]) + si[N: 2*N]
                            wi = st.binom.pmf(k=obs, n=ki, p=delay_rate)
                            wi_sum += wi
                        wi_sum = wi_sum/len(k_samples)
                        w = np.prod(wi_sum)
                        list_sj.append(sj)
                        list_w.append(w)

        # if sum(list_w) == 0:
        # timeout = 1
        tmp_particles = []      #particle reinvigoration
        prev_particles = np.unique(np.array(root.B), axis=0)
        for prev_state in prev_particles:
            tmp_state = np.zeros(len(prev_state), dtype=int)
            check = (prev_state[0:N] + prev_state[N:2*N]) - obs
            if np.all(check >= 0):
                dept_sampled = np.zeros(self.N, dtype=int)

                rem = np.where(obs > prev_state[N:2*N])[0]     # came from prev_state[0:N]
                low = np.where(obs <= prev_state[N:2*N])[0]
                if len(rem) == 0:
                    tmp_state[N:2*N] = np.maximum(0, prev_state[N:2*N] - obs + dept_sampled)
                    tmp_state[0:N] = np.maximum(0, prev_state[0:N] - dept_sampled)
                else:
                    tmp_state[N:2*N][low] = np.maximum(0, prev_state[N:2*N][low] - obs[low] + dept_sampled[low])
                    tmp_state[0:N][low] = np.maximum(0, prev_state[0:N][low] - dept_sampled[low])
                    tmp_state[0:N][rem] = np.maximum(0, (prev_state[0:N][rem] - (obs[rem] - prev_state[N:2*N][rem] -
                                                                                 dept_sampled[rem])))
                    tmp_state[N:2*N][rem] = np.maximum(0, prev_state[N:2*N][rem] - obs[rem] - dept_sampled[rem])

            elif np.any(check >= 0):       # (x+y) - obs < 0 will be zero and the rest solved as above
                neg_check = np.where(check < 0)[0]
                pos_check = np.where(check >= 0)[0]
                tmp_state[0:N][neg_check] = 0
                tmp_state[N:2*N][neg_check] = 0

                # dept_sampled = k_samples[np.random.choice(len(k_samples))]
                dept_sampled = np.zeros(self.N, dtype=int)

                rem = np.where(obs[pos_check] > prev_state[N:2*N][pos_check])[0]     # came from prev_state[0:N]
                low = np.where(obs[pos_check] <= prev_state[N:2*N][pos_check])[0]

                if len(rem) == 0:
                    tmp_state[N:2*N][pos_check] = np.maximum(0, prev_state[N:2*N][pos_check] - obs[pos_check] +
                                                             dept_sampled[pos_check])
                    tmp_state[0:N][pos_check] = np.maximum(0, prev_state[0:N][pos_check] -
                                                           dept_sampled[pos_check])
                else:
                    tmp_state[N:2*N][pos_check][low] = np.maximum(0, prev_state[N:2*N][pos_check][low] -
                                                                obs[pos_check][low] + dept_sampled[pos_check][low])
                    tmp_state[0:N][pos_check][low] = np.maximum(0, prev_state[0:N][pos_check][low] -
                                                                dept_sampled[pos_check][low])
                    tmp_state[0:N][pos_check][rem] = np.maximum(0, (prev_state[0:N][pos_check][rem] -
                                                        (obs[pos_check][rem] - prev_state[N:2*N][pos_check][rem] -
                                                                            dept_sampled[pos_check][rem])))
                    tmp_state[N:2*N][rem] = np.maximum(0, prev_state[N:2*N][pos_check][rem] - obs[pos_check][rem] -
                                                       dept_sampled[pos_check][rem])

            elif np.all(check < 0):     # no zero: (x+y) - obs < 0 : all departed
                tmp_state = np.zeros(len(prev_state), dtype=int)

            tmp_state[action] = min(self.model.queue_length[0], tmp_state[action]+1)
            tmp_state[-N:] = obs
            tmp_particles.append(tmp_state)

            particles_idx = np.random.choice(len(tmp_particles), self.max_particles, replace=True)
            particles = []
            for i in range(self.max_particles):
                particles.append(tmp_particles[particles_idx[i]])

        if sum(list_w) > 0:
            norm_w = list_w / np.sum(list_w)
            if len(list_sj) == 1:
                particles = list_sj * self.max_particles
            else:
                particles_idx = np.random.choice(len(list_sj), self.max_particles, p=norm_w, replace=True)
                for i in range(self.max_particles):
                    particles.append(list_sj[particles_idx[i]])
        else:
            if len(tmp_particles) == 1:
                particles = tmp_particles * self.max_particles
            else:
                particles_idx = np.random.choice(len(tmp_particles), self.max_particles, replace=True)
                for i in range(self.max_particles):
                    particles.append(tmp_particles[particles_idx[i]])

        new_root.B.extend(particles)

        # Prune the tree which can no more be explored
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        self.tree.root.parent = None

        self.model.prev_obs = obs
        self.root_particles = particles



    @staticmethod
    def normalize(arr):
        if isinstance(arr, list):
            arr = np.array(arr)

        return np.divide(arr, np.sum(arr))

    def gen_particle(self, n: int, prob: Union[list, np.ndarray] = None):
        """
        # Apply for non equal queue lengths
        If no probabilty distribution given, do uniform particle generation, else generate based
        on the prob distribution
        :param n: no of particles to be generated
        :param prob: probability distribution for the particles
        :return: list of n particles
        """

        if prob is None or all(p == 0 for p in prob):
            all_particles = np.random.choice(self.model.qns, size=n)
        else:
            all_particles = np.random.choice(self.model.qns, size=n, p=self.normalize(prob))
        return list(all_particles)

    @staticmethod
    def particle_distribution(arr):
        """
        Based on particles in the belief space, generate new probability distributions for belief states
        :param arr: particles
        :return: belief state probability distributions
        """
        cnt = Counter(arr)
        cnt_sum = sum(cnt.values())
        prob = {k: v / cnt_sum for k, v in cnt.items()}
        return prob

    def ucb1(self, node_h):
        """
        get new action based on ucb1 algorithm
        :param node_h: parent node
        :param state: current state
        :return: action that maximises the ucb1 formulation and check of common actions between parent and current leaf
        """
        action = []
        poss_actions = self.model.actions_list
        all_V = np.zeros(len(poss_actions))
        all_nodes = node_h.action_map
        for i in range(len(all_nodes)):
            curr_node = all_nodes[poss_actions[i]]
            V = curr_node.V
            N_h = curr_node.parent.N
            N_ha = curr_node.N

            if N_h == 0:
                ucb_factor = 0.0
            elif N_ha == 0:
                ucb_factor = np.inf
            else:
                ucb_factor = np.sqrt(np.log(N_h) / N_ha)
            v = V + (self.C * ucb_factor)
            all_V[i] = v
            action.append(curr_node.name)

        if np.all(all_V == 0) or (len(set(all_V)) == 1):
            max_V_loc = np.random.choice(range(len(all_V)))

        else:
            max_V_loc = np.where(all_V == max(all_V))[0]
            if len(max_V_loc) > 1:
                max_V_loc = np.random.choice(max_V_loc)

        max_action = action[int(max_V_loc)]

        return max_action


