import os

INPUTS = "./inputs"
OUTPUTS = "./outputs"
TRAIN_DATA_PATH = os.path.join(INPUTS, "aug_train.csv")
TEST_DATA_PATH = os.path.join(INPUTS, "aug_test.csv")

FOLDS = 5
PROBLEM_TYPE = "classification"
TARGET_VARIABLE = "target"

OPTIMIZATION = False  # if True, will use Bayesian optimization with gaussian process to find the best hyperparameters

MODEL_NAME = "xgboost"
MODEL_PATH = "./models"

THRESHOLD = 0.5