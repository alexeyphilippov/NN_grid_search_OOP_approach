from Net_coach import Net_coach
from Net import Net

import time
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

sns.set()



class Grid:
    """
    Notice that you can both search for w1 and w2 independently using grid_1d and search it mutually using grid_2d.

    """

    def __init__(self, x: DataFrame, y: DataFrame) -> None:
        self.__x = x
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

    def grid_1d(self, parameter: str, lower_bound: float, upper_bound: float, n_values: int) -> None:

        scores = []
        if parameter == 'w1':
            self.w1s = np.linspace(lower_bound, upper_bound, n_values).astype(int).tolist()
            for param in self.w1s:
                net = Net([10, param, 8, 1])
                coach = Net_coach(x=self.__x, y=self.__y, model=net)
                scores.append(round(coach.get_error_on_cv(), 2))

        elif parameter == 'w2':
            self.w2s = np.linspace(lower_bound, upper_bound, n_values).astype(int).tolist()
            for param in self.w2s:
                net = Net([10, 3, param, 1])
                coach = Net_coach(x=self.__x, y=self.__y, model=net)
                scores.append(round(coach.get_error_on_cv(), 2))

        elif parameter == 'p':
            self.ps = np.linspace(lower_bound, upper_bound, n_values)
            for param in self.ps:
                net = Net([10, 3, 8, 1], p=param)
                coach = Net_coach(x=self.__x, y=self.__y, model=net)
                scores.append(round(coach.get_error_on_cv(), 2))

        elif parameter == 'lr':
            lrs = np.linspace(lower_bound, upper_bound, n_values)
            for param in lrs:
                net = Net([10, 3, 8, 1])
                coach = Net_coach(x=self.__x, y=self.__y, model=net, lr=param)
                scores.append(round(coach.get_error_on_cv(), 2))

        plt.plot(np.linspace(lower_bound, upper_bound, n_values), scores)
        plt.show()

    """
    grid_2d
    
        Analog of grid_1d method. It commits a mutual search of w1 and w2.
    
    """

    def grid_2d(self, lower_bound1: float, upper_bound1: float, n_values1: int, lower_bound2: float,
                upper_bound2: float, n_values2: int) -> None:

        outer_sc_arr = []

        for w1 in np.linspace(lower_bound1, upper_bound1, n_values1).astype(int).tolist():
            inner_sc_arr = []
            for w2 in np.linspace(lower_bound2, upper_bound2, n_values2).astype(int).tolist():
                net = Net([10, w1, w2, 1])
                coach = Net_coach(x=self.__x, y=self.__y, model=net)
                inner_sc_arr.append(round(coach.get_error_on_cv(), 2))
            outer_sc_arr.append(inner_sc_arr)

        sns.heatmap(outer_sc_arr)
        plt.show()
