import pandas as pd
import os
from IPython.display import display
from Utils import *
from Train import *
from Model import *


def main():
    path = os.getcwd() + '\\Dataset\\Data_Train.xlsx'
    train_df = read_file(path, True)

    train_df = journey(train_df)
    train_df = dropper(train_df, ['Date_of_Journey', 'Route'])

    train_df = durations(train_df)

    train_df = dropper(train_df, ['Additional_Info'])

    train_df = stops(train_df)

    train_df = times(train_df)

    train_airline = encoder(train_df, 'Airline')
    train_source = encoder(train_df, 'Source')
    train_destination = encoder(train_df, 'Destination')
    train_dduration = encoder(train_df, 'Dep_Duration')
    train_aduration = encoder(train_df, 'Arrival_Duration')

    train = pd.concat([train_df, train_airline, train_source,
                       train_destination, train_dduration, train_aduration], axis=1)

    train = dropper(train, ['Airline', 'Source', 'Destination', 'Dep_Duration', 'Arrival_Duration'])

    print('The final dataframe is:')
    display(train.head())

    heatmap(train)

    x = train[['Total_Stops', 'Journey Date', 'Journey Month', 'Journey Day',
               'Journey Weekend', 'Journey Week', 'Duration', 'Airline_Air India',
               'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
               'Airline_Jet Airways Business', 'Airline_Multiple carriers',
               'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
               'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
               'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
               'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
               'Destination_Kolkata', 'Destination_New Delhi',
               'Dep_Duration_Evening_dep', 'Dep_Duration_Morning_dep',
               'Dep_Duration_Night_dep', 'Arrival_Duration_Evening_arr',
               'Arrival_Duration_Morning_arr', 'Arrival_Duration_Night_arr']]

    y = train[['Price']]

    print(f'The shape of x is {x.shape} and y is {y.shape}')

    feature_selector(x,y)

    build_model(x,y,0.2,'Random Forest')




