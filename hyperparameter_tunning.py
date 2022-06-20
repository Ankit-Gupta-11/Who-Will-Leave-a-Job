import pandas as pd    
import numpy as np        
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
from skopt import gp_minimize
from skopt import space
import os

import config
 

def optimize(params, param_names, x, y):
   
    # convert params to dictionary
    params = dict(zip(param_names, params))

    # initialize model with current parameters
    clf = XGBClassifier(tree_method = 'hist', **params)
    
    # initialize stratified k fold
    kf = StratifiedKFold(n_splits = 5)
    
    i = 0
    
    # initialize auc scores list
    auc_scores = []
    
    #loop over all folds
    for index in kf.split(X = x, y = y):
        train_index, test_index = index[0], index[1]
        
        x_train = x.iloc[train_index,:]
        y_train = y[train_index]

        x_test = x.iloc[test_index,:]
        y_test = y[test_index]
        
        #fit model
        clf.fit(x_train, y_train)
        
        y_pred = clf.predict_proba(x_test)
        y_pred_pos = y_pred[:,1]
        
        auc = roc_auc_score(y_test, y_pred_pos)
        print(f'Current parameters of fold number {i} -> {params}')
        print(f'AUC score of test {i} f {auc}')

        i = i+1
        auc_scores.append(auc)
        
    return -1 * np.mean(auc_scores)
    

def get_best_params(clf):    
    #define a parameter space
    param_spaces = [space.Integer(100, 2000, name = 'n_estimators'),
                    space.Real(0.01,100, name = 'min_child_weight'),
                    space.Real(0.01,1000, name = 'gamma'),
                    space.Real(0.1, 1, prior = 'uniform', name = 'colsample_bytree'),
    ]

    # make a list of param names this has to be same order as the search space inside the main function
    param_names = ['n_estimators' ,'min_child_weight', 'gamma', 'colsample_bytree']

    # by using functools partial, i am creating a new function which has same parameters as the optimize function except 
    # for the fact that only one param, i.e. the "params" parameter is required. 
    # This is how gp_minimize expects the optimization function to be. 
    # You can get rid of this by reading data inside the optimize function or by defining the optimize function here.

    df = pd.read_csv(os.path.join(config.OUTPUTS, "aug_train_processed_kfold.csv"))
    X = df.drop(['kfold', config.TARGET_VARIABLE], axis = 1)
    y = df[config.TARGET_VARIABLE]
    optimize_function = partial(optimize, param_names = param_names, x = X, y = y)

    result = gp_minimize(optimize_function, dimensions = param_spaces, n_calls = 5, n_random_starts = 5, verbose = 10)

    best_params = dict(zip(param_names, result.x))

    return best_params
