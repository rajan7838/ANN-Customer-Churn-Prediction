import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import *
from src.exception.custom_exception import CustomException


class DataPreprocessing:

    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

    def fit_transform(self, df, fit=True):
        try:
            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]

            # Encode categorical columns
            for col in CATEGORICAL_COLS:
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    self.encoders[col] = le
                else:
                    X[col] = self.encoders[col].transform(X[col])

            # Scale numerical columns
            if fit:
                X[NUMERICAL_COLS] = self.scaler.fit_transform(X[NUMERICAL_COLS])
            else:
                X[NUMERICAL_COLS] = self.scaler.transform(X[NUMERICAL_COLS])

            return X, y

        except Exception as e:
            raise CustomException(e)

    def save_preprocessor(self):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(
            {"encoders": self.encoders, "scaler": self.scaler},
            self.preprocessor_path
        )

    def load_preprocessor(self):
        data = joblib.load(self.preprocessor_path)
        self.encoders = data["encoders"]
        self.scaler = data["scaler"]





                        
        

