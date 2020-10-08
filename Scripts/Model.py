import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from Utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def plot_residue(actual,predicted):
    plt.figure(figsize=(8, 8))
    sns.distplot(actual - predicted.reshape(-1, 1))
    plt.show()

    return

def model_save(name,model):
    file = open(f'{name}.pkl', 'wb')

    # dump information to that file
    pickle.dump(model, file)

    file.close()

def tuner(x_train,y_train,model_name,model):
    if model_name =='Random Forest':
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15, 100]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 5, 10]

        # Create the random grid

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}

        tune = RandomizedSearchCV(estimator=model,
                                   param_distributions=random_grid, scoring='neg_mean_squared_error',
                                   n_iter=2, cv=2, verbose=2, random_state=42, n_jobs=1)
        tune.fit(x_train,y_train)

        print(tune.best_params_)

        return tune


def build_rforest(xtrain,xtest,ytrain,ytest,tuned=True):
    if tuned:
        regressor = RandomForestRegressor()
        regressor_updated=tuner(xtrain,ytrain,'Random Forest',regressor)
    else:
        regressor_updated = RandomForestRegressor()

    predictions = regressor_updated.predict(xtest)

    plot_residue(ytest,predictions)

    e_mse, e_rmse, e_mae, e_r2, e_agg = metric(ytest, predictions)

    model_save('flight_fare', regressor_updated)

    return e_mse, e_rmse, e_mae, e_r2, e_agg



