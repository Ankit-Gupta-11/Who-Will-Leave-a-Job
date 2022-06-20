import pandas as pd 
import numpy as np  
import os
from sklearn.metrics import roc_auc_score 
import joblib 
import argparse


import config
import model_dispatcher


def run(fold, MODEL_NAME):
    df = pd.read_csv(os.path.join(config.OUTPUTS, "aug_train_processed_kfold.csv"))
    
    df_train = df[df['kfold'] != fold].sample(frac = 1).reset_index(drop = True)
    df_val = df[df['kfold'] == fold].sample(frac = 1).reset_index(drop = True)

    model = model_dispatcher.get_model(MODEL_NAME)

    X_train = df_train.drop(['kfold', config.TARGET_VARIABLE], axis = 1)
    y_train = df_train[config.TARGET_VARIABLE]

    X_val = df_val.drop(['kfold', config.TARGET_VARIABLE], axis = 1)
    y_val = df_val[config.TARGET_VARIABLE]
    
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)

    print(f"FOLD {fold}, {MODEL_NAME}, AUC: {score}")

    # save the model
    joblib.dump(model, 
                os.path.join(config.MODEL_PATH, f"{MODEL_NAME}_{fold}.bin"))

    return score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type = int)

    parser.add_argument("--model", type = str) 
    args = parser.parse_args()

    run(args.fold, args.model)