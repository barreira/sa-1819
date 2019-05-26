import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.layers import LeakyReLU

pd.options.mode.chained_assignment = None  # gets rid of warning in line 26
np.random.seed(9)


def load(file='sales.csv'):
    sales = pd.read_csv(file, header=0, names=['Month', 'Advertising', 'Sales'])
    df = pd.DataFrame(sales)

    return df


def preprocess(df):
    for idx, month in enumerate(df['Month']):
        df['Month'][idx] = month.split('-')[-1]  # e.g. '01-Jan' -> 'Jan'

    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
              'Nov': 11, 'Dec': 12}
    seasons = {'Jan': 1, 'Feb': 1, 'Mar': 1, 'Apr': 2, 'May': 2, 'Jun': 2, 'Jul': 3, 'Aug': 3, 'Sep': 3, 'Oct': 4,
               'Nov': 4, 'Dec': 4}

    month_names = {str(number): month for month, number in months.items()}  # corresponds to months mapping reversed
    season_names = {'1': 'Winter', '2': 'Spring', '3': 'Summer', '4': 'Autumn'}  # number as str for df.rename below

    df['Season'] = df['Month'].map(seasons)  # creates new column 'Season' with the season of each month
    df['Month'] = df['Month'].map(months)  # e.g. 'Jan' -> 1

    df = pd.get_dummies(df, columns=['Month'], prefix='', prefix_sep='')  # one-hot encoding of 'Month'
    df = df.rename(month_names, axis='columns')  # didn't do this from the start to preserve order of months

    df = pd.get_dummies(df, columns=['Season'], prefix='', prefix_sep='')  # one-hot encoding of 'Season'
    df = df.rename(season_names, axis='columns')  # didn't do this from the start to preserve order of seasons

    m = list(months.keys())  # ['Jan', 'Feb', ..., 'Nov', 'Dec']
    s = list(season_names.values())  # ['Winter', 'Spring', 'Summer', 'Autumn']
    df = df[m + s + ['Advertising', 'Sales']]  # reorder columns so as to have 'Advertising' and 'Sales' last

    return df


def describe(df):

    def mkdir(dir_name):
        try:
            os.rmdir(dir_name)
        except OSError:
            pass
        try:
            os.mkdir(dir_name)
        except OSError:
            pass

    # Print dataset description to file

    print(df.describe(), file=open('dataset_description.txt', 'w'))

    # Generate feature distribution graphs to files inside 'dist' folder

    name = 'dist'
    mkdir(name)

    for key in df.keys():
        if key != 'Month':
            df.hist(column=key)
            fig_name = name + '-' + key.lower() + '.png'
            plt.savefig(name + '/' + fig_name)


def visualize():
    df = load()

    print('### Before preprocessing ###')
    print(df.head())

    describe(df)
    df = preprocess(df)

    print('### After preprocessing ###')
    print(df.head())


def train_test_split(df, sliding_window, train_size):
    num_features = len(df.columns)
    sequence_len = sliding_window + 1

    res = []
    for i in range(len(df.values) - sliding_window):
        res.append(df.values[i: i + sequence_len])
    res = np.array(res)  # numpy array of matrices (sliding window across values)

    train = res[:train_size, :]
    x_train = train[:, :-1]  # last value is the last y after the sliding window
    y_train = train[:, -1][:, -1]  # get last value of each matrix corresponding to the label
    x_test = res[train_size:, :-1]
    y_test = res[train_size:, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))

    return [x_train, y_train, x_test, y_test]


def build_model(sliding_window, num_features):
    model = Sequential()
    model.add(LSTM(64, input_shape=(sliding_window, num_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, input_shape=(sliding_window, num_features)))
    model.add(Dropout(0.2))
    model.add(Dense(16))  # default initializer (Glorot Uniform)
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1))  # default initializer (Glorot Uniform)
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    return model


def print_model(model, file):
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)


def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def print_predictions(y_test, predictions):
    diff = []

    for i in range(len(y_test)):
        diff.append(abs(y_test[i] - predictions[i]))
        print('Value: %f ---> Prediction: %f Diff: %f' % (y_test[i], predictions[i], diff[i]))

    plt.plot(y_test, color='blue', label='y_test')
    plt.plot(predictions, color='red', label='prediction')
    # plt.plot(diff, color='green', label='diff')
    plt.legend(loc='upper left')

    plt.show()


def lstm(sliding_window=3, train_size=24):
    """
    :param sliding_window: Number of past months used to make prediction of current month
    :param train_size: How many months should be used for training (the rest will be used for testing)
    """
    df = load()
    df = preprocess(df)
    print('df', df.shape)

    x_train, y_train, x_test, y_test = train_test_split(df[::-1], sliding_window, train_size)
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    model = build_model(sliding_window, len(df.columns))
    history = model.fit(x_train, y_train, batch_size=16, epochs=115, validation_data=(x_test, y_test), verbose=1)

    print_history_loss(history)
    print_model(model, 'lstm_model.png')

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.3f MSE (%.3f RMSE)' % (train_score[0], math.sqrt(train_score[0])))

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.3f MSE (%.3f RMSE)' % (test_score[0], math.sqrt(test_score[0])))
    print(model.metrics_names)

    predictions = model.predict(x_test)
    print_predictions(y_test, np.squeeze(np.asarray(predictions)))


if __name__ == '__main__':
    visualize()
    lstm()
