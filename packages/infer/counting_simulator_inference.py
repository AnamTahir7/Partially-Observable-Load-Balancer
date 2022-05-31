import numpy as np
import packages.simulator.counting_simulator as counting_simulator
import pymc3 as pm
from theano import tensor as T
from packages.utils.plot_utils import plot_mixture_model_fit, plot_gamma_mixture_model_fit, plot_gamma_model_fit
import pickle


# TODO Random seed
# TODO comment verbose
# TODO add verbose options to other inference types
# TODO clean up
# TODO add option to load precomputed results

class InferCountingSimulator:
    """Class for inference of a counting Simulator from inter-arrival and service time data"""

    def __init__(self, epsilon=1e-12, nComponents=None, hyper=None, show_plots=False, save_inferences=False):
        """
        :param epsilon: upper bound on the error of the truncated dirichlet process
        :param hyper: dictonary of hyper parameters
                      hyper['concentration']: 2d np array - hyper parameter for the shape and rate of gamma distributed concentration parameter
                      hyper['stick']: 1d np array - hyper parameter for the beta distributed stick proportion
                      hyper['inv_variance']: 2d np array - hyper parameter for the shape and rate of gamma distributed precision parameter
                      hyper['mean']: 1d np array - hyper parameter for the mean of normal distributed mean parameter
                      hyper['exp_rate']: 2d np array - hyper parameter for the shape and rate of gamma distributed rate parameter of the exponential model
                      hyper['gamma_shape']: 2d np array - hyper parameter for the shape and rate of gamma distributed shape parameter of the gamma model
                      hyper['gamma_rate']: 2d np array - hyper parameter for the shape and rate of gamma distributed rate parameter of the gamma model
        """

        self.hyper = {'concentration': np.array([1, 1]),
                      'stick': np.array([1.]),
                      'inv_variance': np.array([1, 1]),
                      'mean': np.array([0]),
                      'exp_rate': np.array([1., 1.]),
                      'gamma_shape': np.array([1., 1.]),
                      'gamma_rate': np.array([1., 1.])
                      }

        if hyper is not None:
            self.hyper.update(hyper.copy())

        # Number of mixture components with epsilon error
        if nComponents is None:
            self.nComponents = np.ceil(
                2 - self.hyper['concentration'][0] / self.hyper['concentration'][1] * np.log(epsilon)).astype(int)
        else:
            self.nComponents = nComponents

        self.show_plots = show_plots
        self.save_inferences = save_inferences

    # TODO comment
    def get_precomputed_CountingSimulator(self, inter_arrival_times, no_jobs, inference_type='gammaMixture', sim_iat=False):

        if not sim_iat:
            if inference_type == 'gamma' or 'gammaMixture':
                if inter_arrival_times.size == 1:
                    path = '../bin/queuing_datat_19_07_22/results/' + inference_type + '/'
                    extension = '.npz'
                    nServers=8
                    pred_service=[]
                    for num in range(nServers):
                        file_name = inference_type+'_service_' + str(num)
                        all_data = np.load(path+file_name + extension, allow_pickle=True)
                        pred_service.append(all_data['pred_data'].copy())

                    k_samples = [self._leaving_packets_det_arrival_stoch_service(inter_arrival_times, pred_service[server])
                                 for
                                 server in
                                 range(nServers)]
                # Stochastic inter arrival times
                else:
                    # Calculate posterior predictive samples for inter_arrival_times
                    # pred_arrival = inter_arrival_times

                    path = '../network_data/kaggle/arr_srv/' + inference_type + '/'
                    extension = '.npz'

                    file_name = inference_type + '_arrival'
                    iat = np.load(path + file_name + extension, allow_pickle=True)
                    inter_arrival_times = iat['pred_data']
                    rnd_times = np.random.choice(len(inter_arrival_times), size=no_jobs)
                    pred_arrival = inter_arrival_times[rnd_times]


                    nServers = 6
                    pred_service = []
                    for num in range(nServers):
                        file_name = inference_type + '_service_' + str(num)
                        all_data = np.load(path + file_name + extension, allow_pickle=True)
                        pred_service.append(all_data['pred_data'][rnd_times].copy())
                    k_samples = [self._leaving_packets_stoch_arrival_stoch_service(pred_arrival, pred_service[server]) for
                                 server in range(nServers)]

                        # Calculate pmfs from samples
                w = [np.bincount(k_samples[server]) for server in range(nServers)]
                pmfs = [w_i / np.sum(w_i) for w_i in w]

                CountingSimulatorInstance = counting_simulator.Generic(pmfs)
            # else:
            #     raise TypeError('inference type not supported')
        else:
            if inference_type == 'gamma' or 'gammaMixture':
                # Calculate posterior predictive samples for inter_arrival_times
                # pred_arrival = inter_arrival_times

                path = 'network_data/kaggle/gam_arr_gamMix_srv/' + inference_type + '_3/'
                # path = '../network_data/kaggle/gam_arr_gamMix_srv/' + inference_type + '_3/'
                # path = '../network_data/kaggle/arr_srv/' + inference_type + '/'
                extension = '.npz'

                rnd_times = np.random.choice(len(inter_arrival_times), size=no_jobs)
                pred_arrival = inter_arrival_times

                nServers = 20
                pred_service = []
                for num in range(nServers):
                    file_name = inference_type + '_service_' + str(num)
                    all_data = np.load(path + file_name + extension, allow_pickle=True)
                    pred_service.append(all_data['pred_data'][rnd_times].copy())



                k_samples = [self._leaving_packets_stoch_arrival_stoch_service(pred_arrival, pred_service[server]) for
                             server in range(nServers)]

                    # Calculate pmfs from samples
                w = [np.bincount(k_samples[server]) for server in range(nServers)]
                pmfs = [w_i / np.sum(w_i) for w_i in w]

                CountingSimulatorInstance = counting_simulator.Generic(pmfs)
            else:
                raise TypeError('inference type not supported')

        return CountingSimulatorInstance, pred_arrival, pred_service

    def get_only_iat_plots(self, inter_arrival_times, inference_type='logMixture', fig_name=0):
        # Case for mixture of gammas inference
        if inference_type == 'gammaMixture':
        # Stochastic inter arrival times
            # Calculate posterior predictive samples for inter_arrival_times
            pred_arrival = self._pred_gamma_mixture(inter_arrival_times, name=inference_type + '_arrival' + str(fig_name))
        if inference_type == 'gamma':
            # Calculate posterior predictive samples for inter_arrival_times
            pred_arrival = self._pred_gamma(inter_arrival_times, name=inference_type + '_arrival'+ str(fig_name))

        if inference_type == 'postMean':
            mu = [(self.hyper['exp_rate'][0] + data.__len__()) / (self.hyper['exp_rate'][1] + np.sum(data)) for data
                  in
                  inter_arrival_times]

            if self.save_inferences:
                np.savez('postMean', mu=mu, service_times=inter_arrival_times, hyper=self.hyper)


    def get_CountingSimulator(self, inter_arrival_times, service_times, inference_type='logMixture'):
        """
        Returns a counting simulator from data

        :param inter_arrival_times: np.array: if 1 data point --> deterministic arrivals
                                              multiple data points are samples of inter arrival times
        :param service_times: list of np.array: A list of service time datas for each server
        :param inference_type: str - options: 'logMixture' non parametric inference
                                              'param' parametric inference using posterior means
        :return: Instance of a counting Simulator inferred from data
        """

        # cast to np array
        # inter_arrival_times = np.array(inter_arrival_times)

        # Count number of servers
        nServers = service_times.__len__()

        # Cases:
        # logMixture
        # gammaMixture
        # gamma
        # postMean

        # Case for log mixture of normals inference
        if inference_type == 'logMixture':

            # Determenistic inter arrival times
            # if inter_arrival_times.size == 1:
            if inter_arrival_times.__sizeof__()== 1:
                # Calculate posterior predictive samples for all service_times
                pred_service = [
                    self._pred_log_mixture(service_times[server], name=inference_type + '_service_' + str(server)) for
                    server
                    in range(nServers)]
                k_samples = [self._leaving_packets_det_arrival_stoch_service(inter_arrival_times, pred_service[server])
                             for
                             server in
                             range(nServers)]

            # Stochastic inter arrival times
            else:
                # Calculate posterior predictive samples for inter_arrival_times
                pred_arrival = self._pred_log_mixture(inter_arrival_times, name=inference_type + '_arrival')

                # Calculate posterior predictive samples for all service_times
                pred_service = [
                    self._pred_log_mixture(service_times[server], name=inference_type + '_service_' + str(server)) for
                    server
                    in range(nServers)]

                k_samples = [self._leaving_packets_stoch_arrival_stoch_service(pred_arrival, pred_service[server]) for
                             server
                             in
                             range(nServers)]

            # Calculate pmfs from samples
            w = [np.bincount(k_samples[server]) for server in range(nServers)]
            pmfs = [w_i / np.sum(w_i) for w_i in w]

            CountingSimulatorInstance = counting_simulator.Generic(pmfs)

        # Case for mixture of gammas inference
        elif inference_type == 'gammaMixture':
            # Determenistic inter arrival times
            if len(inter_arrival_times) == 1:
                # Calculate posterior predictive samples for all service_times
                pred_service = [
                    self._pred_gamma_mixture(service_times[server][0], name=inference_type + '_service_' + str(server)) for
                    server
                    in range(nServers)]
                k_samples = [self._leaving_packets_det_arrival_stoch_service(inter_arrival_times, pred_service[server])
                             for
                             server in
                             range(nServers)]

            # Stochastic inter arrival times
            else:
                # Calculate posterior predictive samples for inter_arrival_times
                pred_arrival = self._pred_gamma_mixture(inter_arrival_times, name=inference_type + '_arrival')

                # Calculate posterior predictive samples for all service_times
                pred_service = [
                    self._pred_gamma_mixture(service_times[server][0], name=inference_type + '_service_' + str(server)) for
                    server
                    in range(nServers)]

                k_samples = [self._leaving_packets_stoch_arrival_stoch_service(pred_arrival, pred_service[server]) for
                             server
                             in
                             range(nServers)]

            # Calculate pmfs from samples
            w = [np.bincount(k_samples[server]) for server in range(nServers)]
            pmfs = [w_i / np.sum(w_i) for w_i in w]

            CountingSimulatorInstance = counting_simulator.Generic(pmfs)

        # Case for Exponential Service/ Gamma arrivals posterior mean inference
        elif inference_type == 'postMean':

            # Determenistic inter arrival times
            if inter_arrival_times.size == 1:
                mu = [(self.hyper['exp_rate'][0] + data.__len__()) / (self.hyper['exp_rate'][1] + np.sum(data)) for data
                      in
                      service_times]

                if self.save_inferences:
                    np.savez('postMean', mu=mu, service_times=service_times, hyper=self.hyper)

                CountingSimulatorInstance = counting_simulator.DetArrivalExpService(inter_arrival_times, mu)

            # Stochastic inter arrival times
            else:
                mu = [(self.hyper['exp_rate'][0] + data.__len__()) / (self.hyper['exp_rate'][1] + np.sum(data)) for data
                      in
                      service_times]
                alpha_mean, beta_mean = self._mean_gamma_model(inter_arrival_times)

                if self.save_inferences:
                    np.savez('postMean', alpha_mean=alpha_mean, beta_mean=beta_mean, mu=mu, service_times=service_times,
                             hyper=self.hyper)

                CountingSimulatorInstance = counting_simulator.GammaArrivalExpService(alpha_mean, beta_mean, mu)

        # Case for Gamma inference
        elif inference_type == 'gamma':
            # Determenistic inter arrival times
            if len(inter_arrival_times) == 1:
                # Calculate posterior predictive samples for all service_times
                pred_service = [self._pred_gamma(service_times[server][0], name=inference_type + '_service_' + str(server))
                                for server in range(nServers)]
                k_samples = [self._leaving_packets_det_arrival_stoch_service(inter_arrival_times, pred_service[server])
                             for
                             server in
                             range(nServers)]
            # Stochastic inter arrival times
            else:
                # Calculate posterior predictive samples for inter_arrival_times
                pred_arrival = self._pred_gamma(inter_arrival_times, name=inference_type + '_arrival')

                # Calculate posterior predictive samples for all service_times
                pred_service = [self._pred_gamma(service_times[server], name=inference_type + '_service_' + str(server))
                                for server in range(nServers)]

                k_samples = [self._leaving_packets_stoch_arrival_stoch_service(pred_arrival, pred_service[server]) for
                             server in range(nServers)]

            # Calculate pmfs from samples
            w = [np.bincount(k_samples[server]) for server in range(nServers)]
            pmfs = [w_i / np.sum(w_i) for w_i in w]

            CountingSimulatorInstance = counting_simulator.Generic(pmfs)

        else:
            raise TypeError('inference type not supported')

        return CountingSimulatorInstance

    def _pred_log_mixture(self, data, Nsamples=1000, name=None):
        """
        Samples posterior predictive from an infinite mixture model with Gaussian observations
            -tansforms non negative data to log domain -> standardization -> inference -> transforms back
        :param data: np.array: list of non-negative observations
        :param Nsamples: Number of posterior/posterior predictive samples for the inference
        :return: np.array- Posterior predcitive samples
        """
        if name is None:
            name = 'logMixture'

        log_data = np.log(data)
        mu_data = np.mean(log_data)
        var_data = np.var(log_data)
        log_std_data = (log_data - mu_data) / var_data

        # Define pymc3 model (infinite mixture - Gaussian)
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', self.hyper['concentration'][0], self.hyper['concentration'][1])
            beta = pm.Beta('beta', self.hyper['stick'][0], alpha, shape=self.nComponents)
            w = pm.Deterministic('w', stick_breaking(beta))

            tau = pm.Gamma('tau', self.hyper['inv_variance'][0], self.hyper['inv_variance'][1], shape=self.nComponents)
            mu = pm.Normal('mu', self.hyper['mean'][0], tau=tau, shape=self.nComponents)

            components = pm.Normal.dist(mu, tau=tau)
            obs = pm.NormalMixture('obs', w=w, mu=mu, tau=tau, observed=log_std_data)

        # Sample posterior
        with model:
            trace = pm.sample(Nsamples, tune=10000, init='adapt_diag')  # , init='advi')  # random_seed=SEED,

        # Sample posterior predictive
        with model:
            predictive = pm.sample_posterior_predictive(trace)
        pred_data = np.exp((predictive['obs'].T.reshape(-1) + mu_data) * var_data)

        if self.show_plots:
            plot_mixture_model_fit(trace, log_std_data, figname=name)
        if self.save_inferences:
            np.savez(name, trace=trace, predictive=predictive, model=model, log_std_data=log_std_data,
                     log_data=log_data,
                     pred_data=pred_data)

            with open(name + '_trace.pkl', 'wb') as file:
                pickle.dump(trace, file)

        return pred_data

    # TODO comment
    def _pred_gamma_mixture(self, data, Nsamples=1000, name=None):
        """
        Samples posterior predictive from an infinite mixture model with Gaussian observations
            -tansforms non negative data to log domain -> standardization -> inference -> transforms back
        :param data: np.array: list of non-negative observations
        :param Nsamples: Number of posterior/posterior predictive samples for the inference
        :return: np.array- Posterior predcitive samples
        """
        if name is None:
            name = 'gammaMixture'

        # Define pymc3 model (infinite mixture - Gaussian)
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', self.hyper['concentration'][0], self.hyper['concentration'][1])
            beta = pm.Beta('beta', self.hyper['stick'][0], alpha, shape=self.nComponents)
            w = pm.Deterministic('w', stick_breaking(beta))

            alpha_i = pm.Gamma('alpha_i', self.hyper['gamma_shape'][0], self.hyper['gamma_shape'][1],
                               shape=self.nComponents)
            beta_i = pm.Gamma('beta_i', self.hyper['gamma_rate'][0], self.hyper['gamma_rate'][1],
                              shape=self.nComponents)

            components = pm.Gamma.dist(alpha=alpha_i, beta=beta_i)
            obs = pm.Mixture('obs', w=w, comp_dists=components, observed=data)

        # Sample posterior
        with model:
            trace = pm.sample(Nsamples, tune=10000, init='adapt_diag')  # , init='advi')  # random_seed=SEED,

        # Sample posterior predictive
        with model:
            predictive = pm.sample_posterior_predictive(trace)
        pred_data = predictive['obs'].T.reshape(-1)

        if self.show_plots:
            plot_gamma_mixture_model_fit(trace, data, figname=name)
        if self.save_inferences:
            np.savez(name, trace=trace, predictive=predictive, model=model, data=data, pred_data=pred_data)

            with open(name + '_trace.pkl', 'wb') as file:
                pickle.dump(trace, file)
        return pred_data

    # TODO Document
    def _pred_gamma(self, data, Nsamples=1000, name=None):
        """
        Samples posterior predictive from an
        :param data: np.array: list of non-negative observations
        :param Nsamples: Number of posterior/posterior predictive samples for the inference
        :return: np.array- Posterior predcitive samples
        """

        if name is None:
            name = 'gamma'

        # Define pymc3 model (iid Gamma observations)
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', self.hyper['gamma_shape'][0], self.hyper['gamma_shape'][1])
            beta = pm.Gamma('beta', self.hyper['gamma_rate'][0], self.hyper['gamma_rate'][1])

            obs = pm.Gamma('obs', alpha=alpha, beta=beta, observed=data)

        # Sample posterior
        with model:
            trace = pm.sample(Nsamples, tune=10000, init='adapt_diag')

            # Sample posterior predictive
        with model:
            predictive = pm.sample_posterior_predictive(trace)

        pred_data = predictive['obs'].T.reshape(-1)

        if self.show_plots:
            plot_gamma_model_fit(trace, data, figname=name)
        if self.save_inferences:
            np.savez(name, trace=trace, predictive=predictive, model=model, data=data, pred_data=pred_data)

            with open(name + '_trace.pkl', 'wb') as file:
                pickle.dump(trace, file)

        return pred_data

    def _mean_gamma_model(self, data, Nsamples=1000):
        """
        Returns posterior mean for gamma observation model
        :param data: np.array- Gamma distributed observations
        :param Nsamples: int - number of posterior samples
        :return: alpha_mean, beta_mean: Posterior mean estiamtes for shape and rate parameters
        """

        # Define pymc3 model (iid Gamma observations)
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', self.hyper['gamma_shape'][0], self.hyper['gamma_shape'][1])
            beta = pm.Gamma('beta', self.hyper['gamma_rate'][0], self.hyper['gamma_rate'][1])

            obs = pm.Gamma('obs', alpha=alpha, beta=beta, observed=data)

        # Sample posterior
        with model:
            trace = pm.sample(Nsamples, tune=10000, init='adapt_diag')

        # Calculate posterior means
        alpha_mean = trace['alpha'].mean()
        beta_mean = trace['beta'].mean()
        return alpha_mean, beta_mean

    @staticmethod
    def _leaving_packets_stoch_arrival_stoch_service(inter_arrival_times, service_times):
        """
        Calculates samples of leaving packets from queue for given inter-arrival and service times
        :param inter_arrival_times: np.array of inter arrival times
        :param service_times: list of np.arrays of serivce times
        :return: samples of leaving packets
        """
        samples = []
        idx_arrival = 0
        idx_service = 0
        nService = service_times.__len__()
        while True:
            curInterArrival = inter_arrival_times[idx_arrival]
            cum_service_time = 0
            idx_sample = 0
            while cum_service_time < curInterArrival:
                cum_service_time += service_times[idx_service]
                if idx_service == nService - 1:
                    return np.array(samples)
                idx_service += 1
                idx_sample += 1
            samples.append(idx_sample - 1)
            idx_arrival += 1

    @staticmethod
    def _leaving_packets_det_arrival_stoch_service(inter_arrival_time, service_times):
        """
        Calculates samples of leaving packets from queue for given determenistic inter-arrival time and service times
        :param inter_arrival_time: determenistic inter arrival time
        :param service_times: list of np.arrays of service times
        :return: samples of leaving packets
        """
        samples = []
        idx_service = 0
        nService = service_times.__len__()
        while True:
            cum_service_time = 0
            idx_sample = 0
            while cum_service_time < inter_arrival_time:
                cum_service_time += service_times[idx_service]
                if idx_service == nService - 1:
                    return np.array(samples)
                idx_service += 1
                idx_sample += 1
            samples.append(idx_sample - 1)


def stick_breaking(beta):
    portion_remaining = T.concatenate([[1], T.extra_ops.cumprod(1 - beta)[:-1]])
    w = beta * portion_remaining
    w = T.set_subtensor(w[-1], 1 - T.sum(w[:-1]))
    return w
