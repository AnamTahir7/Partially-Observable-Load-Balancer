""" A place to store all utility functions """
import os
from datetime import datetime
import numpy as np
import collections
from typing import Iterable


def create_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def get_datetime():
    """
    Returns current data and time as e.g.: '2019-4-17_21_40_56'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_short_datetime():
    """
    Returns current data and time as e.g.: '0417_214056'
    """
    return datetime.now().strftime("%m%d_%H%M%S")


def ndarray_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    # elif isinstance(x, Iterable) and any(isinstance(y, np.ndarray) for y in x):
    #     return [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
    return x


def recursive_conversion(d, func=ndarray_to_list):
    if isinstance(d, dict):
        return {k: recursive_conversion(v) for k, v in d.items()}
    return func(d)


def numberToBase(n, b, l):
    m = n
    if n == 0:
        state = np.zeros(l, dtype=int)
        # return state
    else:
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        digits = digits[::-1]

        state = np.concatenate([np.zeros(l - len(digits), dtype=int), digits])
    return state


def baseToNumber(arr, b):
    b = int(b)
    if np.sum(arr) == 0:
        return 0
    nz_idx = np.min(np.nonzero(arr))
    nz_arr = arr[nz_idx:]
    z = np.array(list(map(lambda x: pow(b, x), range(len(nz_arr) -1, -1, -1))))
    # z = np.array(list(map(lambda x: pow(b,x), range(len(arr) -1, -1, -1))))
    idx = np.sum(nz_arr * z)
    return idx
