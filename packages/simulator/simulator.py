import numpy as np
from abc import ABC, abstractmethod
from packages.simulator.counting_simulator import CountingSimulator
from packages.simulator.reward_queuing_simulator import RewardQueuingSimulator


class Simulator(ABC):
    """Base class for simulator"""

    @abstractmethod
    def simulate(self, curState, action):
        pass


class QueuingSimulator(Simulator):
    """Base class for queuing simulator"""

    def __init__(self, CountingSimulatorInstance, RewardQueuingSimulatorInstance, BufferLengths,
                 initialBufferFillings=None):
        """

        :param CountingSimulatorInstance: Instance of the CountingSimulator class
        :param RewardQueuingSimulatorInstance: Instance of the CountingSimulator class
        :param BufferLengths: array of ints - Buffer size for each of the parralel queues
        :param initialBufferFillings: array of ints ([0,BufferLength]^nServers) - initial queue fillings
        """

        # Save properties
        self.nServers = BufferLengths.__len__()

        self.BufferLengths = np.array(BufferLengths, dtype=int)
        self.vec2idx_tuple = tuple(self.BufferLengths + 1)
        self.CountingSimulatorInstance = CountingSimulatorInstance
        self.RewardQueuingSimulatorInstance = RewardQueuingSimulatorInstance

        # If buffer filling is none initalize queues as empty
        if initialBufferFillings is None:
            initialBufferFillings = np.zeros_like(self.BufferLengths)

        self.BufferFillings = np.array(initialBufferFillings)

    @property
    def CountingSimulatorInstance(self):
        return self._CountingSimulatorInstance

    @CountingSimulatorInstance.setter
    def CountingSimulatorInstance(self, x):
        try:
            assert isinstance(x, CountingSimulator)
        except AssertionError:
            raise AssertionError('Not a valid counting simulator')

        try:
            assert self.nServers == x.nServers
        except AssertionError:
            raise AssertionError('inconsistent number of servers')

        self._CountingSimulatorInstance = x

    @property
    def RewardQueuingSimulatorInstance(self):
        return self._RewardQueuingSimulatorInstance

    @RewardQueuingSimulatorInstance.setter
    def RewardQueuingSimulatorInstance(self, x):
        try:
            assert isinstance(x, RewardQueuingSimulator)
        except AssertionError:
            raise AssertionError('Not a valid reward simulator')

        try:
            assert np.allclose(self.BufferLengths, x.BufferLengths)
        except AssertionError:
            raise AssertionError('inconsistent number of Buffer lengths')

        self._RewardQueuingSimulatorInstance = x

    @abstractmethod
    def simulate(self, curState, action):
        pass

#
# class ParallelQueuing(QueuingSimulator):
#     def __init__(self, CountingSimulatorInstance, RewardQueuingSimulatorInstance, BufferLengths,
#                  initialBufferFillings=None):
#         """
#
#         :param CountingSimulatorInstance: Instance of the CountingSimulator class
#         :param RewardQueuingSimulatorInstance: Instance of the CountingSimulator class
#         :param BufferLengths: array of ints - Buffer size for each of the parralel queues
#         :param initialBufferFillings: array of ints ([0,BufferLength]^nServers) - initial queue fillings
#         """
#         super().__init__(CountingSimulatorInstance, RewardQueuingSimulatorInstance, BufferLengths,
#                          initialBufferFillings=initialBufferFillings)
#
#     def simulate(self, curState, action):
#         """
#         simulates one time step in the queuing system
#         :param action: server number where the packet is send
#         :param curState: array of ints - current state encoding the buffer fillings for each queue
#         :return: nextState : array of ints - next state encoding the buffer fillings for each queue
#                  observation : int - encoding the buffer fillings for each queue
#                  reward : scalar - reward
#
#         """
#
#         self.BufferFillings = curState  # np.array(np.unravel_index(curState, tuple(self.BufferLengths)))
#         # Simulate departures from the queues
#         k = self.CountingSimulatorInstance.draw()
#
#         b_prime = np.maximum(self.BufferFillings - k, np.zeros(self.nServers))
#         if b_prime[action] < self.BufferLengths[action]:
#             b_prime[action] += 1
#
#         nextState = np.array(b_prime, dtype=int)
#         # No noise
#         observation = np.ravel_multi_index(nextState, self.vec2idx_tuple)  # nextState
#
#         self.BufferFillings = nextState  # np.ravel_multi_index(self.BufferFillings, tuple(self.BufferLengths))
#
#         reward = self.RewardQueuingSimulatorInstance.get_reward(self.BufferFillings, action)
#
#         return nextState, observation, reward


class PODParallelQueuing(QueuingSimulator):
    def __init__(self, CountingSimulatorInstance, RewardQueuingSimulatorInstance, BufferLengths,
                 initialBufferFillings=None,
                 delay_rates=1):
        """

        :param CountingSimulatorInstance: Instance of the CountingSimulator class
        :param RewardQueuingSimulatorInstance: Instance of the CountingSimulator class
        :param BufferLengths: array of ints - Buffer size for each of the parralel queues
        :param initialBufferFillings: array of ints ([0,BufferLength]^nServers) - initial queue fillings
        :param delay_rates: scalar or array of size BufferLengths: encodes the succes probability of a packet feedback
        """
        super().__init__(CountingSimulatorInstance, RewardQueuingSimulatorInstance, BufferLengths,
                         initialBufferFillings=initialBufferFillings)

        if not hasattr(delay_rates, "__len__"):
            delay_rates = delay_rates * np.ones(self.nServers)
        self.delay_rates = delay_rates

        self.FeedbackUnobserved = np.zeros_like(self.BufferFillings)
        self.FeedbackObserved = np.zeros_like(self.BufferFillings)

    def simulate(self, curState, action):
        """
        simulates one time step in the queuing system
        :param action: server number where the packet is send
        :param curState: array of ints - current state encoding the buffer fillings, number of unobserved and observed
                                         packets for each queue
        :return: nextState : array of ints - next state encoding the buffer fillings, number of unobserved and observed
                                             packets for each queue
                 observation : int - encoding the buffer fillings for each queue
                 reward : scalar - reward

        """
        # print(curState)
        self.BufferFillings = curState[:self.nServers]
        self.FeedbackUnobserved = curState[self.nServers:2 * self.nServers]
        self.FeedbackObserved = curState[2 * self.nServers:]

        # Simulate departures from the queues
        k = self.CountingSimulatorInstance.draw()

        # Simulate feedback channel
        newPackets = np.array(np.minimum(self.BufferFillings, k) + self.FeedbackUnobserved, int)
        l = np.random.binomial(newPackets, self.delay_rates)
        self.FeedbackUnobserved = newPackets - l
        self.FeedbackObserved = l

        b_prime = np.maximum(self.BufferFillings - k, np.zeros(self.nServers))

        if b_prime[action] < self.BufferLengths[action]:
            b_prime[action] += 1
        self.BufferFillings = np.array(b_prime, dtype=int)


        observation = self.FeedbackObserved
        nextState_tuple = np.array([self.BufferFillings, self.FeedbackUnobserved, self.FeedbackObserved])
        nextState = np.array([z for b in nextState_tuple for z in b])
        # observation = np.ravel_multi_index(self.FeedbackObserved,self.vec2idx_tuple)#self.FeedbackObserved
        reward = self.RewardQueuingSimulatorInstance.get_reward(self.BufferFillings, self.FeedbackObserved, action)

        return nextState, observation, reward
