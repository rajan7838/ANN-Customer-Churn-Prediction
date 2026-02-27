import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_registry import ModelRegistry


def train_pipeline():

    # Step 1: Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Step 2: Preprocessing
    preprocessor = DataPreprocessing()
    X_train, y_train = preprocessor.fit_transform(train_df, fit=True)
    X_test, y_test = preprocessor.fit_transform(test_df, fit=False)

    preprocessor.save_preprocessor()

    # Step 3: Training
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train, X_test, y_test)

    trainer.save_model()

    # Step 4: Save everything
    registry = ModelRegistry()
    registry.save_model_and_preprocessor(model, preprocessor)

    print("Training Completed Successfully")


if __name__ == "__main__":
    train_pipeline()