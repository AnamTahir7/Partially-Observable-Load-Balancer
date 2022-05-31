import time
import numpy as np
from tqdm import tqdm
from packages.queuing_system.pos.pos_env import ModelExtractor
from packages.queuing_system.pos.solver_pos import SolverCreator
from packages.queuing_system.model import SystemModel
from packages.queuing_system import QueueingSystem

class PosCreator(QueueingSystem):
    """
    Solves the routing problem using MCTS - POS
    """

    def __init__(self, max_play: int, arrival_rate: int, shape, queues_lengths: np.ndarray, dept_rates: np.ndarray,
                 N: int, discount: float, drop_reward: int, starting_st: np.ndarray, delay_rate, *args, srv_times,
                   inter_arr_times):
        """
        Initialise the pocmp instance
        :param max_play: total number of arrivals and departures
        :param arrival_rate: average rate with which an arrival occurs
        :param queues_lengths: buffer size of each queue
        :param dept_rates: service rate of each queue
        :param N: Total number of queues
        :param discount: discount factor
        :param drop_reward: penalty for dropping a packet
        :param starting_st: initial state of the system
        """
        self.T = 10  # Tree depth
        QueueingSystem.__init__(self, max_play, arrival_rate, shape, queues_lengths, dept_rates, N, discount, drop_reward,
                                starting_st, delay_rate, ModelExtractor, 'POL', srv_times, inter_arr_times)

    def run(self, pos_values: SystemModel, queuing_simulator, m, output_dir, pid, no_particles) -> SystemModel:
        """
        Solving the routing problem using queuing_system algorithm till maximum number of plays
        :param pos_values: dict generated in run.py to store results of this method
        :return: number of packet drops, total reward gained, queue activity (arrival and departure), total arrivals
        to the queue
        """

        model: ModelExtractor = self.model_extractor(self.arrival_rate, self.queues_lengths, self.dept_rates,
                                                     self.N, self.drop_reward)



        pos = SolverCreator(model, queuing_simulator.RewardQueuingSimulatorInstance.C, no_particles, self.inter_arr_times)
        curr_s_pos = model.curr_state
        curr_obs_pos = curr_s_pos[-self.N:]

        for i in range(pos.model.num_queues):
            pos_values.q_orig[i + 1].append(int(curr_s_pos[i]))
            pos_values.q_obs[i + 1].append(int(curr_obs_pos[i]))

        for i in tqdm(range(self.max_play), desc=self.algorithm + f' {pid}', position=pid, ncols=40):
            pos.solve(self.T, queuing_simulator, i)
            action_pos, action_idx = pos.get_action(pos.unack_jobs_each_queue)
            pos_values.tot_arr_q[action_idx] += 1   # which queue is the job assigned to
            pos_values.action_list.append(int(action_idx))
            self.delay_sim(pos_values, action_idx, i, model, queuing_simulator)
            for k in range(model.num_queues):
                pos_values.q_orig[k + 1].append(int(model.curr_state[k]))
                pos_values.q_obs[k + 1].append(int(model.curr_obs[k]))
            k_samples = [queuing_simulator.CountingSimulatorInstance.draw() for _ in range(5)]
            pos.update_belief(queuing_simulator, action_pos, model.curr_obs, self.delay_rate, k_samples, i)

        for j in range(len(self.action_list)):
            pos_values.states_list.append(self.state_list[j].tolist())

        return pos_values
