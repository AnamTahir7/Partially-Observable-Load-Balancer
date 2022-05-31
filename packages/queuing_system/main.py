import os
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import numpy as np
import scipy.stats as st

from packages.queuing_system.utils import create_directory, get_short_datetime, recursive_conversion
from packages.queuing_system.pos.gen_pos import PosCreator
from packages.queuing_system.ejmo.gen_ejmo import Ejmo
from packages.queuing_system.jmo.gen_jmo import Jmo
from packages.queuing_system.sed.gen_sed import Sed
from packages.queuing_system.jsq.gen_jsq import Jsq
from packages.queuing_system.d_jsq.gen_d_jsq import D_Jsq
from packages.simulator.simulator import PODParallelQueuing
from packages.simulator.counting_simulator import GammaArrivalExpService, GammaArrivalParetoService
from packages.simulator.reward_queuing_simulator import *
from packages.queuing_system.compile import Compile
from packages.queuing_system.model import SystemModel
from packages.queuing_system.plotting import QueuingResultPlotting
from packages.infer.counting_simulator_inference import InferCountingSimulator



def worker(w: int, p: dict, queuing_simulator, output_dir: str, j: int, delay_rate, alpha, beta, mu, a, m,
           total_threads):
    pid = int(mp.current_process().name.split('-')[1]) - 1
    max_play = p.get('max_play_per_mc')
    N = p.get('N')

    inter_arr_times = st.gamma.rvs(alpha, scale=1 / beta, size=max_play + 1,
                 random_state=np.random.RandomState())
    srv_times = {}
    for i in range(N):
        if p.get('srv') == 'pareto':
            srv_times['q{}'.format(i)] = np.random.pareto(a[i], size=max_play+1) * (m[i])
        elif p.get('srv') == 'exp':
           srv_times['q{}'.format(i)] = st.expon.rvs(scale=1 / mu[i], size=max_play + 1,
                                                           random_state=np.random.RandomState())
        else:
            raise NotImplementedError

    # POS
    queue_pos = PosCreator.from_config(p, j, srv_times, inter_arr_times)
    pos_model = SystemModel(N)
    queue_pos.run(pos_model, queuing_simulator, w, output_dir, pid % total_threads, no_particles=p.get('no_prtcls'))

    # EJMO - with 20% exploration
    queue_ejmo = Ejmo.from_config(p, j, srv_times, inter_arr_times)
    ejmo_model = SystemModel(N)
    queue_ejmo.run(ejmo_model, queuing_simulator)

    # MO
    queue_jmo = Jmo.from_config(p, j, srv_times, inter_arr_times)
    jmo_model = SystemModel(N)
    queue_jmo.run(jmo_model, queuing_simulator)

    # SED
    queue_sed = Sed.from_config(p, j, srv_times, inter_arr_times)
    sed_model = SystemModel(N)
    queue_sed.run(sed_model, queuing_simulator)

    # JSQ
    queue_jsq = Jsq.from_config(p, j, srv_times, inter_arr_times)
    jsq_model = SystemModel(N)
    queue_jsq.run(jsq_model, queuing_simulator)

    # Power-of-d-JSQ
    queue_d_jsq = D_Jsq.from_config(p, j, srv_times, inter_arr_times)
    d_jsq_model = SystemModel(N)
    queue_d_jsq.run(d_jsq_model, queuing_simulator)

    curr_output_json = {
        "Departure_Rates": p.get('dept_rates'),
        "beta": p.get('arrival_rate'),
        "alpha": p.get('arrival_rate_alpha'),
        "pareto_m": p.get('pareto_m'),
        "pareto_a": p.get('pareto_a'),
        "Queue_Lengths": p.get('queues_lengths'),
        "Delay_Prob": delay_rate,
        "No_of_processors": p.get('nr_threads'),
        "Total_MC_sim": p.get('max_play_per_mc'),
        "Start_state": p.get('starting_state'),
        "pol": pos_model.asdict(),
        "ejmo": ejmo_model.asdict(),
        "jmo": jmo_model.asdict(),
        "sed": sed_model.asdict(),
        "jsq": jsq_model.asdict(),
        "djsq": d_jsq_model.asdict()
    }

    curr_output_json = recursive_conversion(curr_output_json)
    with open(os.path.join(output_dir, 'data_{:05}.json'.format(w + 1)), 'w') as json_file:
        json.dump(curr_output_json, json_file, indent=4)


def run(**p):
    """
    Runs the three methods and does routing barr on the algorithms.
    :param p: Arguments from the run function, including input parameters and dicts
    :return: Different calculated results from the methods and saving data and plots in the Results folder
    """

    q = p.get('queues_lengths')
    mu = np.array(p.get('dept_rates'))
    r = p.get('reward_func')
    alpha_all = p.get('arrival_rate_alpha')  # shape
    beta_all = p.get('arrival_rate')  # rate of arrival
    delay_all = p.get('delay_prob')
    N = p.get('N')

    # Create a Results directory at the same level as this script
    script_path = Path(__file__).absolute().parent
    results_dir = script_path.joinpath('Results')
    results_dir.mkdir(exist_ok=True)

    for j in range(len(beta_all)):
        beta = beta_all[j]
        alpha = alpha_all[j]

        run_time = get_short_datetime()
        f_name_beta = 'beta{beta}-t{t}'.format(beta=beta, t=run_time)
        output_dir_beta = results_dir.joinpath(f_name_beta)
        create_directory(output_dir_beta)
        delay_rate = delay_all

        run_time = get_short_datetime()
        f_name = 't{t}-{dept}-d{sdelay}-{r}'.format(t=run_time,
                                                    dept='homo' if
                                                    len(set(mu)) == 1 else 'hetro',
                                                    sdelay=str(delay_rate).replace('.', '_'),
                                                    r=r)
        output_dir: Path = results_dir.joinpath(output_dir_beta).joinpath(f_name)
        output_dir.mkdir(parents=True, exist_ok=True)


        # Choose the inter arrival arrival and service time distribution
        if p.get('arr') == 'exp' and p.get('srv') == 'exp':
            if not p.get('inference'):
                CountingSimulatorInstance = GammaArrivalExpService(alpha, beta, mu)
            else:
                no_jobs = p.get('max_play_per_mc') + 1
                inter_arr_times = st.gamma.rvs(alpha, scale=1 / beta, size=no_jobs, random_state=np.random.RandomState())
                InferenceInstance = InferCountingSimulator()
                CountingSimulatorInstance, inter_arr_times, service_times = \
                    InferenceInstance.get_precomputed_CountingSimulator(inter_arrival_times=inter_arr_times,
                                                                        no_jobs=5000, sim_iat=True)
            a = None
            m = None
        elif p.get('arr') == 'gamma' and p.get('srv') == 'pareto':
            if not p.get('inference'):
                CountingSimulatorInstance = GammaArrivalExpService(alpha, beta, mu)
            else:
                no_jobs = p.get('max_play_per_mc') + 1
                inter_arr_times = st.gamma.rvs(alpha, scale=1 / beta, size=no_jobs, random_state=np.random.RandomState())
                InferenceInstance = InferCountingSimulator()
                CountingSimulatorInstance, inter_arr_times, service_times = \
                    InferenceInstance.get_precomputed_CountingSimulator(inter_arrival_times=inter_arr_times,
                                                                        no_jobs=5000, sim_iat=True)
            a = np.ones(N) * p.get('pareto_a')
            m = np.ones(N) * p.get('pareto_m')
        else:
            raise KeyError("undefined distributions")

        # Choose the reward function
        if r == 'pkt_drop':
            RewardSimulatorInstance = SmallQueueNoDropping(q, mu)
        elif r == 'entropy':
            RewardSimulatorInstance = EntropyReward(q, mu)
        elif r == 'exponential':
            RewardSimulatorInstance = ExponentialReward(q, mu)
        elif r == 'linear':
            RewardSimulatorInstance = LinearReward(q, mu)
        elif r == 'self_clk':
            RewardSimulatorInstance = SelfClocking(q, mu)
        elif r == 'prop_alloc':
            RewardSimulatorInstance = PropAllocation(q, mu)
        elif r == 'sed':
            RewardSimulatorInstance = SedR(q, mu)
        elif r == 'variance':
            RewardSimulatorInstance = VarianceReward(q, mu)
        else:
            raise KeyError("undefined reward function")

        QueuingSimulator = PODParallelQueuing(CountingSimulatorInstance, RewardSimulatorInstance, q,
                                              delay_rates=delay_rate)

        nr_threads = p.get('nr_threads', 1)
        _worker = partial(worker, p=p, queuing_simulator=QueuingSimulator, output_dir=output_dir, j=j,
                          delay_rate=delay_rate, alpha=alpha, beta=beta, mu=mu, a=a, m=m, total_threads=nr_threads)

        with mp.Pool(processes=nr_threads, maxtasksperchild=1) as pool:
            pool.map(_worker, range(p.get('mc_sims')), chunksize=1)

        avg_compiler = Compile(output_dir, expected_iter=p.get('mc_sims'))
        avg = avg_compiler.compile().stats()

        print('avg done')
        plotter = QueuingResultPlotting(avg, output_dir, pomcp_compare=False)
        plotter.draw_plots()


