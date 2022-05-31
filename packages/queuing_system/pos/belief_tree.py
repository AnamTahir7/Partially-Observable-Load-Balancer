import numpy as np
from typing import Union

# Performs functions related to the tree - adding, removing a node, updating a node

class Node(object):
    def __init__(self, nid: int, name: str, h: list, parent=None, V: float=0, N: int=0):
        """
        Node initialisation in tree
        :param nid: id for the node
        :param name: name of the node
        :param h: prior history
        :param parent: parent if any of the node
        :param V: Value function of the node, can be initialised with prior belief
        :param N: number of times node is encountered
        """
        self.h = h
        self.V = V
        self.N = N
        self.id = nid
        self.name = name
        self.parent = parent
        self.children = []


class ActionNode(Node):
    # represents the node associated with a POMDP action
    def __init__(self, nid: int, name: Union[int, str], h: list, action_index: str,  parent=None, V: float=0, N: int=0):
        """
        initialise an action node
        :param nid: id for node in tree
        :param name: name of node
        :param h: history
        :param action_index: index of action
        :param parent: parent node
        :param V: value
        :param N: number of times this action is encountered
        """
        Node.__init__(self, nid, name, h, parent, V, N)
        self.action = action_index
        self.obs_map = {}

    def get_child(self, observation):
        """
        get child of the node
        :param observation: current observation
        :return: child or None
        """
        return self.obs_map.get(observation, None)

    def add_child(self, node):
        """
        add node as a child to the action node
        :param node: node to be added
        :return: updated tree
        """
        self.children.append(node)
        obs = str(node.observation)
        self.obs_map[obs] = node


class BeliefNode(Node):
    """
    Represents a node that holds the belief distribution given its history sequence in a belief tree.
    It also holds the received observation after which the belief is updated accordingly
    """

    def __init__(self, nid: int, name: Union[int, str], h: list, obs_index: np.ndarray, parent=None, V: float=0.0,
                 N: int=0):

        """
        initialise a belief node
        :param nid: id
        :param name: name
        :param h: history
        :param obs_index: observation
        :param parent: parent action node
        :param V: value
        :param N: number of times encountered
        """
        Node.__init__(self, nid, name, h, parent, V, N)
        self.observation = obs_index
        self.B = []  # initialise particle list
        self.action_map = {}

    def add_particle(self, particle: list):
        """
        add particle to the belief space
        :param particle: particle to be added
        :return: updated partcile list
        """
        self.B.extend(particle)

    def add_child(self, node):
        """
        add action child
        :param node:
        :return: updated tree
        """
        self.children.append(node)
        self.action_map[node.action] = node

    def get_child(self, action: str):
        """
        get child of the node
        :param observation: current observation
        :return: child or None
        """
        return self.action_map.get(action, None)

    def sample_state(self, model):
        """
        sample a random state from the particles in the current belief space
        :return: sampled state
        """

        rand_idx = np.random.choice(len(self.B))
        state = self.B[rand_idx]
        return state, rand_idx


class BeliefTree:
    # Generate a new node in the tree
    def __init__(self, root_particles: list, obs):
        self.counter = 0
        self.nodes = {}
        self.root = self.add(h=[], name='root', particle=root_particles)

    def add(self, h: list, name: Union[int, str], parent=None, action=None, observation=None, particle: list=None):
        """
         :param h: history sequence
         :param parent: either ActionNode or BeliefNode
         :param action: action name
         :param observation: observation name
         :param particle: new node's particle set
         :param name: name of node
        """

        history = h[:]
        # Instantiate action or belief Node
        if action is not None:
            n = ActionNode(self.counter, name, history, parent=parent, action_index=action)
        else:
            n = BeliefNode(self.counter, name, history, parent=parent, obs_index=observation)

        if particle is not None:
            n.add_particle(particle)

        # add node to belief tree
        self.nodes[n.id] = n
        self.counter += 1

        # register node as parent's child
        if parent is not None:
            parent.add_child(n)
        return n

    def find_or_create(self, h: list, **kwargs):
        """
        find the history in root nodes or create it
        :param h: history
        :param kwargs: name, parent, observation
        :return: node with updated history
        """
        current = self.root
        h_len = len(h)
        root_history_len = len(self.root.h)

        for step in range(root_history_len, h_len):
            current = current.get_child(str(h[step]))           #current_child if it exists
            if current is None:
                return self.add(h, **kwargs)
        return current

    def prune(self, node, exclude=None):
        """
        Removes the entire subtree subscribed to node with exceptions.
        :param node: root of the subtree to be removed
        :param exclude: exception component
        """
        for child in node.children:
            if exclude and exclude.id != child.id:
                self.prune(child, exclude)

        self.nodes[node.id] = None
        del self.nodes[node.id]
