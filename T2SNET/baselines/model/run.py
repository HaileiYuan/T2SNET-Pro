from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

import time
import argparse
import math
import random
from dataload import split_and_norm_data_time, Scaler
from T2SNET_Model import T2SNT
from sklearn.preprocessing import StandardScaler


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='T2SNET', help='dataset name')
parser.add_argument('--cuda', type=int, default=0,  help='which gpu card used')
parser.add_argument('--P', type=int, default=24, help='history steps')
parser.add_argument('--Q', type=int, default=1, help='prediction steps')
parser.add_argument('--d', type=int, default=64, help='dims of outputs')
parser.add_argument('--train_ratio', type=float, default=0.8, help='training set [default : 0.8]')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set [default : 0.2]')
parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--data_file', default='/Users/zouguojian/RF-LSTM-CEEMDAN/Dataset/UnivDorm_Prince.csv', help='traffic file')
parser.add_argument('--model_file', default='PEMS-1', help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)', help='log file')
args = parser.parse_args()

energy_elements = 9
weather_elements = 10

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')
log = open(args.log_file, 'w')
log_string(log, "loading data....")

train_X, train_Y, train_S, test_X, test_Y, test_S, scaler= split_and_norm_data_time(url_univdorm=args.data_file,
                                                                              train_rate=args.train_ratio,
                                                                              valid_rate=args.val_ratio,
                                                                              recent_prior=args.P,
                                                                              pre_len=args.Q)
log_string(log, "loading end....")

# seed = 2
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# np.random.seed(seed)
# random.seed(seed)

def res(model, test_X, test_Y, test_S, scaler, index):
    model.eval()  # 评估模式, 这会关闭dropout
    num_val = test_X.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                energy_X = torch.from_numpy(test_X[start_idx: end_idx, :, index:index+1]).float().to(device)
                weather_X = torch.from_numpy(test_X[start_idx: end_idx, :, energy_elements+1:energy_elements+weather_elements]).float().to(device)
                sparse_X = torch.from_numpy(test_S[start_idx: end_idx]).float().to(device)
                y = test_Y[start_idx: end_idx, index:index+1]
                y_hat = model(energy_input=energy_X, weather_input=weather_X, time_input=sparse_X)
                pred.append(y_hat.cpu().numpy())
                label.append(y)
    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)
    # print(np.concatenate([scaler.inverse_transform(pred), scaler.inverse_transform(label)],axis=1))
    mae, rmse, mape = metric(scaler.inverse_transform(pred, index=index), scaler.inverse_transform(label, index=index))
    return mae, rmse, mape, scaler.inverse_transform(pred,index=index), scaler.inverse_transform(label, index=index)


def train(model, train_X, train_Y, train_S, val_X, val_Y, val_S, scaler, index):
    num_train = train_X.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 43, 46],
                                                        gamma=0.2)
    patient=0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        train_X = train_X[permutation]
        train_Y = train_Y[permutation]
        train_S = train_S[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            energy_X = torch.from_numpy(train_X[start_idx: end_idx, :, index:index+1]).float().to(device)
            weather_X = torch.from_numpy(train_X[start_idx: end_idx, :, energy_elements+1:energy_elements+weather_elements]).float().to(device)
            sparse_X = torch.from_numpy(train_S[start_idx: end_idx]).float().to(device)
            y = torch.from_numpy(train_Y[start_idx: end_idx, index:index+1]).float().to(device)
            optimizer.zero_grad()
            y_hat = model(energy_input=energy_X, weather_input=weather_X, time_input=sparse_X)
            # print(y.shape, y_hat.shape)
            loss = _compute_loss(y, y_hat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_l_sum += loss.cpu().item()
            n += y.shape[0]
            batch_count += 1
        log_string(log, 'in the training step, epoch %d, lr %.6f, loss %.4f, time %.1f sec' % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape, _, _ = res(model, val_X, val_Y, val_S, scaler,index)
        lr_scheduler.step()
        if patient>10: break
        else:patient+=1
        if mae < min_loss:
            patient=0
            log_string(log, 'in the %dth epoch, the validate average loss value is : %.3f' % (epoch+1, mae))
            min_loss = mae
            torch.save(model, args.model_file)


def test(test_X, test_Y, test_S, scaler, index):
    model = torch.load(args.model_file)
    mae, rmse, mape, pred, label = res(model, test_X, test_Y, test_S, scaler,index)
    log_string(log, 'in the test phase,  mae: %.4f, rmse: %.4f, mape: %.6f' % (mae, rmse, mape))
    return pred, label

def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

import numpy

if __name__ == '__main__':
    log_string(log, "model constructed begin....")
    model = T2SNT(energy_in_channel=1,
                  weather_in_channel=weather_elements,
                  time_in_channel=31,
                  seq_len=args.P,
                  pre_len=args.Q,
                  emb_size=args.d,
                  device=device).to(device)
    log_string(log, "model constructed end....")

    pres, labels = [], []

    # scaler_X= StandardScaler()
    # scaler_Y=StandardScaler()
    # trainX = scaler_X.fit_transform(train_X[:, :, 1])
    # trainY = scaler_Y.fit_transform(train_Y[:, 1:2]).ravel()
    # testX = scaler_X.fit_transform(test_X[:, :, 1])

    # from sklearn.ensemble import RandomForestRegressor
    # grid = RandomForestRegressor(max_features=8)
    # grid.fit(train_X[:, :, 1], train_Y[:, 1:2])
    # pred = grid.predict(test_X[:, :, 1])
    # pres.append(scaler.inverse_transform(np.reshape(pred,[-1,1]),index=1))

    for i in range(1,energy_elements):
        print(i, train_Y.shape)
        # scaler_X = StandardScaler()
        # scaler_Y = StandardScaler()
        # trainX = scaler_X.fit_transform(train_X[:, :, i])
        # trainY = scaler_Y.fit_transform(train_Y[:, i:i+1]).ravel()
        # testX = scaler_X.fit_transform(test_X[:, :, i])
        # testY = scaler_Y.fit_transform(test_Y[:, i:i+1])
        # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # import tensorflow as tf
        # tf.random.set_random_seed(1234)
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.LSTM(units=64, input_shape=(trainX.shape[1], trainX.shape[2])))
        # model.add(tf.keras.layers.Dense(1))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # model.compile(loss='mse', optimizer=optimizer)
        #
        # model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)
        # y_pred = model.predict(testX)
        # print(y_pred.shape)
        # pred = scaler.inverse_transform(y_pred,index=i)
        # pres.append(pred)


        # print('the %d -th element'%i)
        log_string(log, "train begin....")
        train(model, train_X, train_Y, train_S, test_X, test_Y, test_S, scaler, index=i)
        log_string(log, "train end....")
        pred, label = test(test_X, test_Y, test_S, scaler,index=i)
        pres.append(pred)
        labels.append(test_Y)

    predicted = np.sum(np.concatenate(pres, axis=1),axis=1)
    observed = scaler.inverse_transform(test_Y[:,0],index=0)
    for i in range(predicted.shape[0]):
        print(predicted[i], observed[i])
    np.savez_compressed('Lab', **{'prediction': predicted, 'truth': observed})
    mae, rmse, mape = metric(predicted, observed)
    log_string(log, 'final results,  average mae: %.4f, rmse: %.4f, mape: %.6f' % (mae, rmse, mape))
