from typing import List
from oop.class_Net_coach import Net_coach
from oop.class_Net import Net

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

sns.set()

List_with_tensors = List[torch.Tensor]
Tensor_List = List[List_with_tensors]


class Grid:
    """
    Notice that you can both search for w1 and w2 independently using grid_1d and search it mutually using grid_2d.

    """

    def __init__(self, X: DataFrame, y: DataFrame):
        self.__X = X
        self.__y = y

    """
    grid_1d

        Searches for best parameter on a grid given with [lower_bound, upper_bound] 
        and step (upper_bound - lower_bound) / n_values

    parameters: 
        parameter: str - a parameter to be inspected
        lower_bound: float
        upper_bound: float
        n_values: int
    """

    def grid_1d(self, parameter, lower_bound, upper_bound, n_values) -> None:

        __scores_arr = []
        __curr_time = time.time()
        __scores = []
        if parameter == 'w1':
            self.w1s = np.linspace(lower_bound, upper_bound, n_values).astype(int).tolist()
            for param in self.w1s:
                __net = Net([10, param, 8, 1])
                __coach = Net_coach(X=self.__X, y=self.__y, model=__net)
                __scores.append(round(__coach.get_error_on_cv(), 2))

        elif parameter == 'w2':
            self.w2s = np.linspace(lower_bound, upper_bound, n_values).astype(int).tolist()
            for param in self.w2s:
                __net = Net([10, 3, param, 1])
                __coach = Net_coach(X=self.__X, y=self.__y, model=__net)
                __scores.append(round(__coach.get_error_on_cv(), 2))

        elif parameter == 'p':
            self.ps = np.linspace(lower_bound, upper_bound, n_values)
            for param in self.ps:
                __net = Net([10, 3, 8, 1], p=param)
                __coach = Net_coach(X=self.__X, y=self.__y, model=__net)
                __scores.append(round(__coach.get_error_on_cv(), 2))

        elif parameter == 'lr':
            __lrs = np.linspace(lower_bound, upper_bound, n_values)
            for param in __lrs:
                __net = Net([10, 3, 8, 1])
                __coach = Net_coach(X=self.__X, y=self.__y, model=__net, lr=param)
                __scores.append(round(__coach.get_error_on_cv(), 2))

        plt.plot(np.linspace(lower_bound, upper_bound, n_values), __scores)
        plt.show()

    """
    grid_2d
    
        Analog of grid_1d method. It commits a mutual search of w1 and w2.
    
    """

    def grid_2d(self, lower_bound1, upper_bound1, n_values1, lower_bound2, upper_bound2, n_values2) -> None:
        __scores = []
        __curr_time = time.time()
        __outer_sc_arr = []

        for w1 in np.linspace(lower_bound1, upper_bound1, n_values1).astype(int).tolist():
            __inner_sc_arr = []
            for w2 in np.linspace(lower_bound2, upper_bound2, n_values2).astype(int).tolist():
                __net = Net([10, w1, w2, 1])
                __coach = Net_coach(X=self.__X, y=self.__y, model=__net)
                __inner_sc_arr.append(round(__coach.get_error_on_cv(), 2))
            __outer_sc_arr.append(__inner_sc_arr)

        sns.heatmap(__outer_sc_arr)
        plt.show()
