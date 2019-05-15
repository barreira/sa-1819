import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

np.random.seed(9)
pd.options.mode.chained_assignment = None  # gets rid of warning on line 28


def load(file_name='sales.csv'):
    sales = pd.read_csv(file_name, header=0, names=['Month', 'Advertising', 'Sales'])
    df = pd.DataFrame(sales)

    return df


def preprocess(df):

    # 'Month' + 'Season'

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

    # Normalize 'Advertising' and 'Sales' values

    scaler = MinMaxScaler()
    df[['Advertising', 'Sales']] = scaler.fit_transform(df[['Advertising', 'Sales']])  # values to interval [0..1]

    return df


def visualize():
    df = load()
    print('### Before preprocessing ###')
    print(df.head(), '\n')
    df = preprocess(df)
    print('### After preprocessing ###')
    print(df.head(), '\n')


def train_test_split(df_dados, janela):
    qt_atributos = len(df_dados.columns)
    mat_dados = df_dados.as_matrix()  # converter dataframe para matriz (lista com lista de cada registo)
    tam_sequencia = janela + 1

    res = []
    for i in range(len(mat_dados) - tam_sequencia):  # numero de registos - tamanho da sequencia
        res.append(mat_dados[i: i + tam_sequencia])
    res = np.array(res)  # dá como resultado um np com uma lista de matrizes (janela deslizante ao longo da serie)

    train = res[-12:, :]
    x_train = train[:, :-1]  # menos um registo pois o ultimo registo é o registo a seguir à janela
    y_train = train[:, -1][:, -1]  # para ir buscar o último atributo para a lista dos labels
    x_test = res[:24, :-1]
    y_test = res[:24:, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], qt_atributos))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], qt_atributos))

    return [x_train, y_train, x_test, y_test]


def build_model(janela):
    model = Sequential()
    model.add(LSTM(64, input_shape=(janela, 3), return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(32, input_shape=(janela, 3), return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def print_model(model, file):
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)


def print_series_prediction(y_test, predic):
    diff = []
    ratio = []

    for i in range(len(y_test)):
        ratio.append((y_test[i] / predic[i]) - 1)
        diff.append(abs(y_test[i] - predic[i]))
        print('valor: %f ---> Previsão: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], ratio[i]))

    plt.plot(y_test, color='blue', label='y_test')
    plt.plot(predic, color='red', label='prediction')  # este deu uma linha em branco
    plt.plot(diff, color='green', label='diff')
    plt.plot(ratio, color='yellow', label='ratio')
    plt.legend(loc='upper left')

    plt.show()


def lstm():
    df = load()
    df = preprocess(df)
    print("df", df.shape)

    sliding_window = 12
    x_train, y_train, x_test, y_test = train_test_split(df[::-1], sliding_window)  # o df[::-1] é o df por ordem inversa
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    model = build_model(sliding_window)
    model.fit(x_train, y_train, batch_size=512, epochs=500, validation_split=0.1, verbose=1)
    print_model(model, "lstm_model.png")

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score[0], math.sqrt(train_score[0])))

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score[0], math.sqrt(test_score[0])))
    print(model.metrics_names)

    p = model.predict(x_test)
    predic = np.squeeze(np.asarray(p))  # transformar uma matriz de uma coluna e n linhas num np array de n elementos
    print_series_prediction(y_test, predic)


if __name__ == '__main__':
    visualize()
    lstm()
