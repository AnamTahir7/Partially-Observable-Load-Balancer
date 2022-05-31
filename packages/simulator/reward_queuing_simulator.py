import numpy as np
from abc import ABC, abstractmethod
from scipy import stats as st
import math

__all__ = ['RewardQueuingSimulator', 'SmallQueueNoDropping', 'EntropyReward', 'ExponentialReward', 'LinearReward',
           'SelfClocking', 'PropAllocation', 'SedR', 'VarianceReward']


class RewardQueuingSimulator(ABC):
    """
    Base class for reward generation
    """

    @abstractmethod
    def __init__(self, BufferLengths, mu):
        self.BufferLengths = np.array(BufferLengths, dtype=int)
        self.mu = mu
        self.nServers = BufferLengths.__len__()
        self.C = 0
        self.dropreward = 1e6

    @abstractmethod
    def get_reward(self, BufferFillings, AcksRcvd, action):
        pass


class SmallQueueNoDropping(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = sum(self.BufferLengths) + self.dropreward

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """

        reward = -np.sum(BufferFillings)

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        # number of empty queues after taking action
        k = len(np.where(BufferFillings == 0)[0])
        reward -= k

        return int(reward)


class EntropyReward(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = st.entropy(self.BufferLengths) + self.dropreward

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """

        reward = st.entropy(BufferFillings)
        if math.isnan(reward):
            reward = 0

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        return reward


class VarianceReward(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = np.var([0, self.BufferLengths[0]]) + self.dropreward

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """

        reward = np.var(BufferFillings)
        if math.isnan(reward):
            reward = 0

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        return reward



class ExponentialReward(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = np.sum(np.power(2, self.BufferLengths)) + self.dropreward

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """

        reward = -np.sum(np.power(2, BufferFillings))

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        # number of empty queues after taking action
        # k = len(np.where(BufferFillings == 0)[0])
        # reward -= k

        return int(reward)


class LinearReward(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = sum(self.BufferLengths) + self.dropreward
        # self.C = sum(self.BufferLengths) + self.dropreward + (5 * len(self.BufferLengths))

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """

        reward = -np.sum(BufferFillings)

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        # number of empty queues after taking action
        # k = len(np.where(BufferFillings == 0)[0])
        # reward -= k

        return int(reward)

class SelfClocking(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = sum(self.BufferLengths)

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """
        penalty = np.sum(AcksRcvd * (BufferFillings == self.BufferLengths).astype(int))
        reward = np.sum(AcksRcvd) - (0.5*penalty)

        return reward

class PropAllocation(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = self.dropreward + np.sum(self.BufferLengths/self.mu)

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """
        reward = -np.sum(BufferFillings / self.mu)

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        if BufferFillings[action] == self.BufferLengths[action]:
            reward += -self.dropreward

        # number of empty queues after taking action
        # k = len(np.where(BufferFillings == 0)[0])
        # reward -= k

        return reward


class SedR(RewardQueuingSimulator):
    """
    Class for queue size cost and packet drops
    """

    def __init__(self, BufferLengths, mu):
        super().__init__(BufferLengths, mu)
        self.C = np.sum(self.BufferLengths/self.mu)

    def get_reward(self, BufferFillings, AcksRcvd, action):
        """
        returns the reward for chosen buffer filling and action
        :param BufferFillings: array of ints - current state encoding the buffer fillings for each queue
        :param action: server number where the packet is send
        :return: scalar - reward
        """
        reward = -np.sum(BufferFillings / self.mu)

        # If it is full give a punishment (maybe extend state space by one dummy state to explicitly model drops)
        # if BufferFillings[action] == self.BufferLengths[action]:
        #     reward += -self.dropreward

        # number of empty queues after taking action
        # k = len(np.where(BufferFillings == 0)[0])
        # reward -= k

        return reward
