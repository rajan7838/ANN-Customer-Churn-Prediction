import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.config import *
from src.logging.logger import logger
from src.exception.custom_exception import CustomException


class DataIngestion:

    def initiate_data_ingestion(self):
        try:
            logger.info("Reading dataset")

            df = pd.read_csv(DATA_FILE)

            # Drop unnecessary columns
            df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

            train, test = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            train_path = os.path.join(ARTIFACTS_DIR, "train.csv")
            test_path = os.path.join(ARTIFACTS_DIR, "test.csv")

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Data ingestion completed")

            return train_path, test_path

        except Exception as e:
            raise CustomException(e)



