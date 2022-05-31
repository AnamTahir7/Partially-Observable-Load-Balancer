import numpy as np
import argparse
import packages.queuing_system.main as main
import packages.queuing_system.main_infer as main_infer

def run(**kwargs):
    """
    Takes input from the user and uses them to generate dictionaries for each method (queuing_system, jmo, ejmo) based on
    these parameters
    :param kwargs: parameters given by the user in the configuration settings
    :return: dict for each method
    """

    all_N = [50]
    offered_load = 0.90
    kwargs['arr'] = 'exp'   # or gamma
    kwargs['srv'] = 'exp'   # or pareto
    buffer_size = 10
    kwargs['inference'] = False  # for section 5.4
    for j in range(len(all_N)):     # Do for every N config in all_N
        kwargs['N'] = all_N[j]
        N = kwargs['N']
        half = int(N / 2)
        kwargs['queues_lengths'] = np.array([buffer_size] * kwargs['N'], dtype=int)
        if (kwargs['arr'] == 'exp' and kwargs['srv'] == 'exp'):
            kwargs['arrival_rate_alpha'] = [1]
            kwargs['dept_rates'] = np.concatenate((np.array([0.2]*half, dtype=float), np.array([0.1]*half, dtype=float)))
            arr_rate = offered_load * sum(kwargs['dept_rates'])
            kwargs['arrival_rate'] = [arr_rate]
        elif (kwargs['arr'] == 'gamma' and kwargs['srv'] == 'pareto'):
            kwargs['arrival_rate_alpha'] = [2]
            kwargs['pareto_a'] = [kwargs['arrival_rate_alpha'][0]]
            kwargs['arrival_rate'] = 12.67
            m1 = np.ones(25) * 6
            m2 = np.ones(25) * 12
            kwargs['pareto_m'] = np.concatenate((m1, m2))
        else:
            raise KeyError("undefined distributions")
        kwargs['max_play_per_mc'] = 5
        kwargs['mc_sims'] = 1
        kwargs['delay_prob'] = [0.6]
        kwargs['no_prtcls'] = 100
        kwargs['nr_threads'] = 1
        kwargs['starting_state'] = 0        #if starting empty
        kwargs['reward_func'] = 'linear'
        assert len(kwargs['dept_rates']) == kwargs['N']
        if kwargs['inference']:
            main_infer.run(**kwargs)
        else:
            main.run(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(**vars(args))
