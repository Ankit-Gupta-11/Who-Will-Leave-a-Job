from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import config
import hyperparameter_tunning


def get_model(model):

    models = {"xgboost": XGBClassifier(),
                "logistic_regression": LogisticRegression(),
                "random_forest": RandomForestClassifier()}


    if config.OPTIMIZATION == True:
        
        best_params = hyperparameter_tunning.get_best_params(models[model])
        model = models[model](**best_params)

        return model 
    
    elif config.OPTIMIZATION == False:
        model = models[model]
        return model



