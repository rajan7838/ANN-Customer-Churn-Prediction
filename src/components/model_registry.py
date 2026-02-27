import mlflow
import mlflow.tensorflow
import joblib
import os
from src.config import MODELS_DIR, ARTIFACTS_DIR
from src.logging.logger import logger

class ModelRegistry:
    """
    Handles saving/loading of the model and preprocessor locally.
    """
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "ann_model.h5")
        self.preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

    def save_model_and_preprocessor(self, model, preprocessor):
        """Save model (as .h5) and preprocessor (as .pkl) locally."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save(self.model_path)
        joblib.dump(preprocessor, self.preprocessor_path)
        logger.info("Model and preprocessor saved locally.")

    def load_model_and_preprocessor(self):
        """Load model and preprocessor from local files (for API)."""
        import tensorflow as tf
        model = tf.keras.models.load_model(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        logger.info("Model and preprocessor loaded.")
        return model, preprocessor

