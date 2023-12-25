import numpy as np
import sys
import os
import pandas as pd
from PyEMD import CEEMDAN
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Scaler:
    def __init__(self, train, n, m):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D)
        """
        self.n=n
        self.m=m
        self.std = np.std(train[:,:n+m], axis=0)
        self.mean = np.mean(train[:,:n+m], axis=0)

        # print(self.std, self.mean)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D)

        Returns:
            {np.ndarray} -- shape(T, D)
        """

        data[:,:self.n+self.m] = (data[:,:self.n+self.m] - self.mean) / (self.std)
        return data

    def inverse_transform(self, data, index):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D)

        Returns:
            {np.ndarray} --  shape (T, D)
        """
        return data * self.std[index] + self.mean[index]

def dataset_building(data, his_len, pre_len, energy_num, weather_num):
    '''

    :param data:
    :param his_len: 24
    :param pre_len: 1
    :return:
    '''
    X, Y, S =[], [], []
    for i in range(len(data) - his_len - pre_len - 1):
        label = data[i + his_len, :energy_num]
        feature = data[i: i + his_len, :energy_num+weather_num]  # 获取输入特征, 包括能耗和气象数据
        feature = np.concatenate([feature, data[i + his_len-24: i + his_len + pre_len-24, :energy_num+weather_num]],axis=0)
        X.append(feature)  # 形成数据集
        Y.append(label)    # 形成label集
        S.append(data[i: i + his_len + pre_len, energy_num+weather_num:])  # 形成输入稀疏变量, 包括天气情况和时间变量
    return X, Y, S

def string_onehot_data(data_string):
    '''

    'Conditions':string,
    'seasonal': string

    :param condition_list: {element: index}
    :param seasonal_list: {element: index}
    :param week: {element: index}
    :param hour: {element: index}
    :return:
    '''
    data = []
    for element in data_string:
        week_zeros = np.zeros(shape=[7])
        hour_zeros = np.zeros(shape=[24])
        week_zeros[element[0]-1] = 1
        hour_zeros[element[1]-1] = 1
        data.append(np.concatenate([week_zeros, hour_zeros],axis=0))
    data = np.array(data)
    return data

def split_and_norm_data_time(url_univdorm,
                             train_rate=0.8,
                             valid_rate=0.2,
                             recent_prior=3,
                             pre_len=1):

    univdorm = pd.read_csv(url_univdorm)
    data_univdorm = univdorm[(univdorm['timestamp'] > '2015-03-01') & (univdorm['timestamp'] < '2015-06-01')].interpolate(method='linear', limit_direction='backward')
    all_energy_float = data_univdorm['energy'].values
    all_weather_float = data_univdorm['Humidity'].values
    all_data_string = data_univdorm[['weekday', 'hour']].values

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMF_energy = emd(all_energy_float).T
    IMF_weather = emd(all_weather_float).T
    print(IMF_energy.shape, IMF_weather.shape)
    data = string_onehot_data(all_data_string)
    all_data = np.concatenate([np.reshape(all_energy_float,[-1, 1]), IMF_energy, np.reshape(all_weather_float,[-1,1]), IMF_weather, data], axis=1) # energy 1, weather 2, onehot 31, shape is (2207, 54)
    del data
    num_of_time, channel = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time)  # 训练集和验证集的上线
    # 数据集构建
    scaler = Scaler(all_data, IMF_energy.shape[1]+1, IMF_weather.shape[1]+1)
    train_X, train_Y, train_S = dataset_building(scaler.transform(all_data[:train_line]), recent_prior, pre_len, IMF_energy.shape[1]+1, IMF_weather.shape[1]+1)
    val_X, val_Y, val_S = dataset_building(scaler.transform(all_data[train_line:]), recent_prior, pre_len, IMF_energy.shape[1]+1, IMF_weather.shape[1]+1)
    print(np.array(train_X).shape, np.array(val_X).shape, np.array(train_Y).shape, np.array(val_Y).shape, np.array(train_S).shape, np.array(val_S).shape)
    return (np.array(train_X), np.array(train_Y), np.array(train_S),
            np.array(val_X), np.array(val_Y), np.array(val_S),scaler)


# univdorm = 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20UnivClass_Abby.csv'
#
# train_X, train_Y, train_S, val_X, val_Y, val_S, test_X, test_Y, test_S, _=split_and_norm_data_time(url_univdorm=univdorm, train_rate=0.8, valid_rate=0.2, recent_prior=24, pre_len=1)
#
# print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape, train_S.shape, test_S.shape)
#
# print(train_Y)
# print(test_Y)