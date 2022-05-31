""" Structure for saving the variables for all the systems """

from dataclasses import dataclass, field, InitVar
from collections import defaultdict
from typing import List, DefaultDict


@dataclass
class SystemModel:
    N: InitVar[int]
    reward: list = field(default_factory=list)
    pkt_drp: list = field(default_factory=list)
    q_orig: DefaultDict[int, List] = field(default_factory=lambda: defaultdict(list))
    q_obs: DefaultDict[int, List] = field(default_factory=lambda: defaultdict(list))
    tot_arr_q: list = field(default_factory=list)
    delay: list = field(default_factory=list)
    action_list: list = field(default_factory=list)
    bstate_map_list: list = field(default_factory=list)
    states_list: list = field(default_factory=list)
    b1_orig: list = field(default_factory=list)
    b2_orig: list = field(default_factory=list)

    def __post_init__(self, N):
        self.tot_arr_q = [0] * N

    def asdict(self):
        return self.__dict__
