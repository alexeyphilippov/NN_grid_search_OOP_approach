from Net import Net
from sklearn.model_selection import KFold
from typing import List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import seaborn as sns

sns.set()

List_with_tensors = List[torch.Tensor]
Tensor_List = List[List_with_tensors]


class Net_coach():

    def __init__(self, x: DataFrame, y: DataFrame, model: Net, n_folds: int = 5, lr: float = 0.75) -> None:

        self.__x: DataFrame
        self.__y: DataFrame
        self.__model: Net
        self.__n_folds: int
        self.__lr = lr

        self.__x = x
        self.__y = y
        self.__model = model
        self.__n_folds = n_folds
        self.__lr: float

    def train(self, x_train: torch.Tensor, x_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor) -> [list,
                                                                                                                  list,
                                                                                                                  list,
                                                                                                                  float]:
        self._is_trained = True

        self.__model.apply(self.__init_weights)

        self.__criterion = torch.nn.MSELoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__lr)

        self.__epo = []
        self.__los_train = []
        self.__los_test = []
        self.__losss_test = []

        N_EPOCHS = 100
        N_LAST_VALS = 20  # Number of last values of RMSE that you look at when calculating mean RMSE
        for epoch in range(N_EPOCHS):
            y_pred = self.__model(x_train)
            y_pred_test = self.__model(x_test)

            loss_train = self.__criterion(y_pred, y_train)
            lossse_test = mean_squared_error(y_pred_test.tolist(), y_test.tolist())
            loss_test = self.__criterion(y_pred_test, y_test)

            self.__optimizer.zero_grad()
            loss_train.backward()
            self.__optimizer.step()

            self.__epo.append(epoch)
            self.__los_train.append(loss_train.item())
            self.__los_test.append(loss_test.item())
            self.__losss_test.append(lossse_test)

        return self.__epo, self.__los_train, self.__losss_test, np.sqrt(np.mean(self.__losss_test[-N_LAST_VALS:]))

    def __get_cv_(self) -> Tensor_List:

        kf = KFold(self.__n_folds)

        train_indexes = []
        test_indexes = []
        for train_index, test_index in kf.split(self.__x):
            train_indexes.append(train_index)
            test_indexes.append(test_index)

        x_train_tensors = [torch.tensor(self.__x.iloc[tr].values).type(torch.FloatTensor) for tr in train_indexes]
        x_test_tensors = [torch.tensor(self.__x.iloc[te].values).type(torch.FloatTensor) for te in test_indexes]
        y_train_tensors = [torch.tensor(self.__y.iloc[tr].values).type(torch.FloatTensor) for tr in train_indexes]
        y_test_tensors = [torch.tensor(self.__y.iloc[te].values).type(torch.FloatTensor) for te in test_indexes]

        cv: List[List[torch.Tensor]] = []
        for i in range(self.__n_folds):
            cv.append([x_train_tensors[i], y_train_tensors[i], x_test_tensors[i], y_test_tensors[i]])

        return cv

    """
    __get_error_on_cv

        Splits this_X and this_y into train and test parts randomly  in order to build a viable cross-validation. Then
        for each split it creates a Net_coach, calls method train and gets results, witch are to be averaged further.

    """

    def get_error_on_cv(self) -> np.ndarray:

        cv = self.__get_cv_()
        errors_on_cv = []
        for x_train, y_train, x_test, y_test in cv:
            ep, los_tr, los_te, arr = self.train(x_train, x_test, y_train, y_test)
            errors_on_cv.append(arr)

        return np.mean(errors_on_cv)

    @staticmethod
    def __init_weights(model) -> None:  # Very often exploding gradient appears, witch leads to inf solutions.
        if type(model) == nn.Linear:  # Despite batch-normalization weights initialization is requiered
            torch.nn.init.xavier_normal_(model.weight)
            model.bias.data.fill_(0.01)

    def get_training_plot(self) -> None:

        if not self._is_trained:
            raise Exception('Call .train method first.')

        else:
            _, ax = plt.subplots()
            ax.plot(self.__epo, self.__los_train, label='train')
            ax.plot(self.__epo, self.__losss_test, label='test')
            ax.legend(loc='upper center', shadow=True, fontsize='x-large')
