

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(ROOT_DIR, "data", "customer_churn.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TARGET_COLUMN = "Exited"
CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERICAL_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 5
 

