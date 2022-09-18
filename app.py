import pandas as pd  
import numpy as np  
import os     
import joblib
import config
import preprocessing


from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates', static_folder='static')

def inference_preprocessing(df_pre):
    # Handling Categorical Variables
    df_pre1 = preprocessing.handle_categorical_variables(df_pre)

    # Handling missing values
    # knn_imputer = joblib.load(os.path.join(config.MODEL_PATH, f"knn_imputer.bin"))
    # X = np.round(knn_imputer.fit_transform(df_pre))
    # df_pre = pd.DataFrame(X, columns = df_pre.columns)

    return df_pre1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    df_pre = pd.read_csv(os.path.join(config.OUTPUTS, "aug_train_processed_kfold.csv"))
    df_pre.drop(df_pre.index, inplace=True)
    df_pre.drop(['kfold', 'enrollee_id', config.TARGET_VARIABLE], axis = 1, inplace = True)

    features_dict = request.form.to_dict()

    for key, value in features_dict.items():

        if key == "training_hours" or key == "city_development_index":
            df_pre.loc[0, key] = float(value)
        else:
            df_pre.loc[0, key] = value

    

    df_pre = inference_preprocessing(df_pre)
    probs = []
    for fold in range(config.FOLDS):

        # Loading the model
        model = joblib.load(os.path.join("models", f"{config.MODEL_NAME}_{fold}.bin"))
        # prob = model.predict(df_pre)
        prob = model.predict_proba(df_pre.to_numpy())[:, 1]
        probs.append(prob)

    # Predicting the target
    pred = np.mean(probs)

    if pred >= config.THRESHOLD: 
        prediction = "Yes, Employee may leave the company"
        return render_template('index.html', prediction_text= prediction)
    
    else:
        prediction = "No, Employee may not Leave the Company"
        return render_template('index.html', prediction_text= prediction)

if __name__=='__main__':
    app.run(debug = True)
    
