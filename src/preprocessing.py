import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import os
import joblib

import config


def handle_categorical_variables(df_pre):
    
    # Making Dictionaries of ordinal features
    gender_map = {
            'Female': 2,
            'Male': 1,
            'Other': 0
            }

    relevent_experience_map = {
        'Has relevent experience':  1,
        'No relevent experience':    0
    }

    enrolled_university_map = {
        'no_enrollment'   :  0,
        'Full time course':    1, 
        'Part time course':    2 
    }
        
    education_level_map = {
        'Primary School' :    0,
        'Graduate'       :    2,
        'Masters'        :    3, 
        'High School'    :    1, 
        'Phd'            :    4
        } 
        
    major_map ={ 
        'STEM'                   :    0,
        'Business Degree'        :    1, 
        'Arts'                   :    2, 
        'Humanities'             :    3, 
        'No Major'               :    4, 
        'Other'                  :    5 
    }
        
    experience_map = {
        '<1'      :    0,
        '1'       :    1, 
        '2'       :    2, 
        '3'       :    3, 
        '4'       :    4, 
        '5'       :    5,
        '6'       :    6,
        '7'       :    7,
        '8'       :    8, 
        '9'       :    9, 
        '10'      :    10, 
        '11'      :    11,
        '12'      :    12,
        '13'      :    13, 
        '14'      :    14, 
        '15'      :    15, 
        '16'      :    16,
        '17'      :    17,
        '18'      :    18,
        '19'      :    19, 
        '20'      :    20, 
        '>20'     :    21
    } 
        
    company_type_map = {
        'Pvt Ltd'               :    0,
        'Funded Startup'        :    1, 
        'Early Stage Startup'   :    2, 
        'Other'                 :    3, 
        'Public Sector'         :    4, 
        'NGO'                   :    5
    }

    company_size_map = {
        '<10'          :    0,
        '10/49'        :    1, 
        '100-500'      :    2, 
        '1000-4999'    :    3, 
        '10000+'       :    4, 
        '50-99'        :    5, 
        '500-999'      :    6, 
        '5000-9999'    :    7
    }
        
    last_new_job_map = {
        'never'        :    0,
        '1'            :    1, 
        '2'            :    2, 
        '3'            :    3, 
        '4'            :    4, 
        '>4'           :    5
    }

    # Transforming Categorical features into numarical features

    df_pre.loc[:,'education_level'] = df_pre['education_level'].map(education_level_map)
    df_pre.loc[:,'company_size'] = df_pre['company_size'].map(company_size_map)
    df_pre.loc[:,'company_type'] = df_pre['company_type'].map(company_type_map)
    df_pre.loc[:,'last_new_job'] = df_pre['last_new_job'].map(last_new_job_map)
    df_pre.loc[:,'major_discipline'] = df_pre['major_discipline'].map(major_map)
    df_pre.loc[:,'enrolled_university'] = df_pre['enrolled_university'].map(enrolled_university_map)
    df_pre.loc[:,'relevent_experience'] = df_pre['relevent_experience'].map(relevent_experience_map)
    df_pre.loc[:,'gender'] = df_pre['gender'].map(gender_map)
    df_pre.loc[:,'experience'] = df_pre['experience'].map(experience_map)

    #encoding city feature using label encoder
    lb_en = LabelEncoder()

    df_pre.loc[:,'city'] = lb_en.fit_transform(df_pre.loc[:,'city']) 

    return df_pre



def handle_missing_values(df_pre):
    
    missing_cols = df_pre.columns[df_pre.isna().any()].tolist()
        
    #dataframe having features with missing values
    df_missing = df_pre[['enrollee_id'] + missing_cols]

    #dataframe having features without missing values
    df_non_missing = df_pre.drop(missing_cols, axis = 1)

    #k-Nearest Neighbour Imputation
    knn_imputer = KNNImputer(n_neighbors = 3)

    X = np.round(knn_imputer.fit_transform(df_missing))
    #Rounding them because these are categorical features

    df_missing = pd.DataFrame(X, columns = df_missing.columns)

    #now lets join both dataframes 
    df_pre2 = pd.merge(df_missing, df_non_missing, on = 'enrollee_id')

    joblib.dump(knn_imputer, 
                os.path.join(config.MODEL_PATH, f"knn_imputer.bin"))

    return df_pre2

def preprocessing(df_pre):
    df_pre = handle_categorical_variables(df_pre)
    df_pre = handle_missing_values(df_pre)
    return df_pre


if __name__ == '__main__':
        
    # Loading the training data
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)

    # Loading the testing data
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    #lets combine train and test sets to preprocess the data

    #First I suggest to create a fake target feature in test set with some same value for every single element
    #By this it will be easy for us to combine and seprate our training and test data after data preprocessing
    #can plot count plot for more intution

    df_test['target'] = -1 #remeber that we have to drop this column later

    df_pre = pd.concat([df_train, df_test], axis = 0).sample(frac = 1).reset_index(drop = True)
    # Just a Tip always reset the indices whenever you join or disjoin two or more datasets

    df_pre2 = preprocessing(df_pre)

    '''
    If you remember i did concatenation between train and test data before preprocessing. 
    Now after preprocessing of data we can seprate train and test data
    '''

    train = df_pre2[df_pre2['target'] != -1].reset_index(drop = True)
    test = df_pre2[df_pre2['target'] == -1].reset_index(drop = True)

    # drop fake target feature from test data 
    test = test.drop('target', axis = 1)
        
    train.to_csv(os.path.join(config.OUTPUTS, "aug_train_processed.csv"), index = False)
    test.to_csv(os.path.join(config.OUTPUTS, "aug_test_processed.csv"), index = False)
