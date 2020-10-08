import pandas as pd
import numpy as np
# from IPython.display import display
import holidays
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

def read_file(path, dropna=True):
    p = path.replace('\\', '/')
    df = pd.read_excel(p)
    if dropna:
        df.dropna(inplace=True)
    # display(df.head())
    # display(df.info())
    # display('The shape of the DataFrame is:', df.shape)
    return df


def dropper(df, col_name):
    df_temp = df.drop(col_name, axis=1)
    return df_temp


def journey(df):
    df_temp = df.copy(deep=True)
    india_holidays = holidays.India()
    df_temp[['Date_of_Journey']] = pd.to_datetime(df_temp['Date_of_Journey'])
    df_temp['Holiday'] = (df_temp['Date_of_Journey'].isin(india_holidays)).astype(int)

    if len(list(df_temp['Holiday'].unique())) < 2:
        df_temp = dropper(df_temp, 'Holiday')
        print('There are no Holidays!!!')

    df_temp['Journey Date'] = df_temp['Date_of_Journey'].dt.day
    df_temp['Journey Month'] = df_temp['Date_of_Journey'].dt.month
    #     df_temp['Journey Year']=df_temp['Date_of_Journey'].dt.year
    df_temp['Journey Day'] = df_temp['Date_of_Journey'].dt.dayofweek
    df_temp['Journey Weekend'] = np.where(((df_temp['Date_of_Journey']).dt.dayofweek) > 5, 1, 0)
    df_temp['Journey Week'] = df_temp['Date_of_Journey'].dt.week

    # display(df_temp.head())

    return df_temp


def durations(df):
    df_temp = df.copy(deep=True)
    duration = list(df_temp.Duration)
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if 'h' in duration[i]:
                duration[i] = duration[i].strip() + ' 0m'
            else:
                duration[i] = '0h ' + duration[i].strip()

    duration_hour = []
    duration_min = []

    for i in range(len(duration)):
        duration_hour.append(duration[i].split(sep='h')[0])
        duration_min.append(duration[i].split(sep='m')[0].split()[-1])

    df_temp = dropper(df_temp, ['Duration'])

    df_temp['Duration_hour'] = duration_hour
    df_temp['Duration_min'] = duration_min

    df_temp['Duration_min_scaled'] = df_temp['Duration_min'].astype(float) / 60  # converting it into the hours
    df_temp['Duration'] = df_temp['Duration_hour'].astype(float) + df_temp['Duration_min_scaled']

    df_temp = dropper(df_temp, ['Duration_min_scaled', 'Duration_hour', 'Duration_min'])

    # display(df_temp.head())

    return df_temp


def stops(df):
    df_temp = df.copy(deep=True)
    stops = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df_temp['Total_Stops'] = df_temp['Total_Stops'].map(stops)

    # display(df_temp.head())

    return df_temp


def timing(x):
    if (x > 7) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 17):
        return 'Afternoon'
    elif (x > 17) and (x <= 20):
        return 'Evening'
    elif (x > 20) or (x <= 7):
        return 'Night'


def times(df):
    df_temp = df.copy(deep=True)

    df_temp['Dep_Time'] = pd.to_datetime(df_temp['Dep_Time'], format='%H:%M')
    df_temp['Arrival_Time1'] = df_temp['Arrival_Time'].apply(lambda x: x.split()[0])
    df_temp['Arrival_Time1'] = pd.to_datetime(df_temp['Arrival_Time1'], format='%H:%M')

    df_temp['Dep_Duration'] = df_temp['Dep_Time'].dt.hour.apply(timing) + '_dep'
    df_temp['Arrival_Duration'] = df_temp['Arrival_Time1'].dt.hour.apply(timing) + '_arr'

    df_temp = dropper(df_temp, ['Arrival_Time1', 'Dep_Time', 'Arrival_Time'])
    # display(df_temp.head())

    return df_temp


def encoder(df, name):
    source = df[[name]]
    df_temp = pd.get_dummies(source, drop_first=True)
    return df_temp


def metric(actual, predicted):
    e_mse = mse(actual, predicted)
    e_mae = mae(actual, predicted)
    e_r2 = r2(actual, predicted)
    e_agm = ((sqrt(e_mse) + e_mae) / 2) * (1 - e_r2)

    return e_mse, sqrt(e_mse), e_mae, e_r2, e_agm

def heatmap(dataframe):
    plt.figure(figsize=(30, 18))
    sns.heatmap(dataframe.corr(), annot=True, cmap="RdYlGn")
    plt.show()

    return

def feature_selector(x,y):
    selector = ExtraTreesRegressor()
    selector.fit(x, y)

    # plot graph of feature importances for better visualization
    plt.figure(figsize=(12, 8))
    feat_importances = pd.Series(selector.feature_importances_, index=x.columns)
    feat_importances.sort_values().plot(kind='barh')
    plt.show()

