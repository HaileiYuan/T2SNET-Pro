import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
import warnings
import datetime
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        print('mae, rmse, mape is : ', mae, rmse, mape)
    return mae, rmse, mape

def plot(ts):
    results = adfuller(ts)
    results_str = 'ADF test, p-value is: {}'.format(results[1])

    grid = plt.GridSpec(2, 2)
    ax1 = plt.subplot(grid[0, :])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])

    ax1.plot(ts)
    ax1.set_title(results_str)
    plot_acf(ts, lags=int(len(ts) / 2 - 1), ax=ax2)
    plot_pacf(ts, lags=int(len(ts) / 2 - 1), ax=ax3)
    plt.show()

# 1. read the original data from database and observe the results of visualization
url_univdorm= 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20PrimClass_Jaden.csv'
univdorm= pd.read_csv(url_univdorm)
data_univdorm= univdorm[(univdorm['timestamp'] > '2015-03-01') & (univdorm['timestamp'] < '2015-06-01')]
dfs_univdorm=data_univdorm['energy']

# 2. visualization
print('Visual display, including stationarity test, autocorrelation, and partial autocorrelation plots')
# If p is less than 0.05, why no difference is needed?
# It means that the data is relatively stable, but seasonal difference is needed.
# plot(dfs_univdorm.values[:500])

def find_pq(ts, d=0, max_p=5, max_q=5):
    best_p, best_q = 0, 0
    best_aic = np.inf

    for p in range(max_p):
        for q in range(max_q):
            model = ARIMA(ts, order=(p, d, q)).fit()
            aic = model.aic

            if aic < best_aic:
                best_aic = aic
                best_p = p
                best_q = q

    return best_p, best_q, best_aic


def version_arima_with_manual(ts, mean, std):
    """
    ARIMA（手动季节差分）
    """
    # 周期大小
    periods = 24

    # 季节差分
    ts_diff = ts - ts.shift(periods)
    # 再次差分（季节差分后p值小于0.05-接近，可认为平稳，若要严格一点也可再做一次差分）
    # ts_diff = ts_diff - ts_diff.shift(1)

    # （训练数据中不能有缺失值，这里差分后前几个值为nan，故去除）
    # ts_diff = ts_diff[~pd.isnull(ts_diff)]

    # 数据拆分
    train, test = train_test_split(ts_diff, train_size=0.8)
    labels = ts.values[train.shape[0]:]

    # 模型训练（训练数据为差分后的数据-已平稳，所以d=0）
    p, q, _ = find_pq(train)
    model = ARIMA(train, order=(p, 0, q)).fit()
    print(model.summary())

    # 拟合结果
    fitted = model.fittedvalues

    # 模型预测
    fcst = model.forecast(test.shape[0])

    # 差分还原（拟合结果）
    fitted += ts.shift(periods)

    # 差分还原（预测结果）
    tmp = ts.loc[train.index].values.tolist() + fcst.values.tolist()
    for i in range(len(tmp) - fcst.shape[0], len(tmp)):
        tmp[i] += tmp[i - periods]
    fcst.loc[:] = tmp[-fcst.shape[0]:]

    # 模型评估
    metric(pred=fcst.values * std + mean, label=labels * std + mean)

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.plot(labels * std + mean, label='Observed', color='black')
    plt.plot(fcst.values * std + mean, label='SARIMA', color='red')
    plt.ylabel("Traffic flow", font1)
    plt.title("Monitoring station 43", font1)
    plt.legend()
    plt.grid(True)
    plt.show()


mean, std =dfs_univdorm.mean(axis=0), dfs_univdorm.std(axis=0)
dfs_univdorm = dfs_univdorm.apply(lambda x: (x - mean)/std)
version_arima_with_manual(dfs_univdorm, mean, std)