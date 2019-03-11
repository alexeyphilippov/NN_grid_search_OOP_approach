import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns

sns.set()


class Net(nn.Module):

    def __init__(self, weights: list = [10, 3, 8, 1], p: float = 0.5) -> None:
        super(Net, self).__init__()

        self.__p = p
        self.__flag = True
        self.__weights = weights

        self.__layer0 = nn.Linear(self.__weights[0], self.__weights[1])
        self.__bn0 = nn.BatchNorm1d(self.__weights[1])

        self.__layer1 = torch.nn.Linear(self.__weights[1], self.__weights[2])
        self.__bn1 = nn.BatchNorm1d(self.__weights[2])

        self.__layer2 = torch.nn.Linear(self.__weights[2], self.__weights[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.__bn0(self.__layer0(x)))

        x = F.relu(self.__bn1(self.__layer1(F.dropout(x, self.__p))))

        x = self.__layer2(F.dropout(x, self.__p))
        return x

    def set_drop_out(self, p: float) -> None:  # Enables to change dropout while training
        self.__p = p

    def use_drop_out(self, flag: bool = True) -> None:  # Turn on/off dropout
        self.__flag = flag
        if not self.__flag:
            self.__p = 0
