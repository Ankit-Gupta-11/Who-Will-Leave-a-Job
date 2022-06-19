import pandas as pd  
import numpy as np  
import os     
import joblib

import config
import preprocessing


def inference_preprocessing(df_pre):
    # Handling Categorical Variables
    df_pre = preprocessing.handle_categorical_variables(df_pre)

    # Handling missing values
    knn_imputer = joblib.load(os.path.join(config.MODEL_PATH, f"knn_imputer.bin"))
    X = np.round(knn_imputer.fit_transform(df_pre))
    df_pre = pd.DataFrame(X, columns = df_pre.columns)

    return df_pre

def predict(df_pre):

    probs = []
    for fold in config.FOLDS:

        # Loading the model
        model = joblib.load(os.path.join(config.MODEL_PATH, f"{config.MODEL_NAME}_{fold}.bin"))
        prob = model.predict_proba(df_pre)[:, 1]
        probs.append(prob)

    # Predicting the target
    pred = np.mean(probs)

    if pred >= config.THRESHOLD: 
        print("Yes, Employee will leave the company")
    
    else:
        print("No, Employee will Stay")


if __name__ == '__main__':
    cols = ['enrollee_id', 'gender', 'enrolled_university', 'education_level',      
       'major_discipline', 'experience', 'company_size', 'company_type',       
       'last_new_job', 'city', 'city_development_index', 'relevent_experience',
       'training_hours']


##################################################### Work in Progress ###############################################
# Deployment...