from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import config
import os

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(config.OUTPUTS, "aug_train_processed.csv")).sample(frac = 1).reset_index(drop = True)
    df['kfold'] = -1

    kf = StratifiedKFold(n_splits = config.FOLDS, shuffle = True, random_state = 11)

    if config.PROBLEM_TYPE == "classification":
        
        for f, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df[config.TARGET_VARIABLE])):
            df.loc[val_idx, 'kfold'] = f

    elif config.PROBLEM_TYPE == "regression":

        # create the bins of the regression data so that stratifiedKFold can be used

        num_bins = np.floor( 1 + np.log2(len(df)))
        df.loc[:, 'bins'] = pd.cut(df[config.TARGET_VARIABLE], bins = num_bins, labels = False)

        for f, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df['bins'])):
            df.loc[val_idx, 'kfold'] = f

        df = df.drop('bins', axis = 1)


    df.to_csv(os.path.join(config.OUTPUTS, "aug_train_processed_kfold.csv"), index = False)