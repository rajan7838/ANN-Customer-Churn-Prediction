import tensorflow as tf
import os
import mlflow
import mlflow.tensorflow
from src.config import *
from src.exception.custom_exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "ann_model.h5")
        self.model = None

    def build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        try:
            input_dim = X_train.shape[1]
            self.model = self.build_model(input_dim)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )

            # --- MLflow logging ---
            mlflow.set_experiment("ANN_Customer_Churn")
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params({
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "dropout_rate": DROPOUT_RATE,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE
                })

                # Train the model
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stop],
                    verbose=1
                )

                # Log final metrics
                mlflow.log_metrics({
                    "train_loss": history.history['loss'][-1],
                    "train_accuracy": history.history['accuracy'][-1],
                    "val_loss": history.history['val_loss'][-1],
                    "val_accuracy": history.history['val_accuracy'][-1]
                })

                # Log the model
                mlflow.tensorflow.log_model(self.model, "model")

            return self.model

        except Exception as e:
            raise CustomException(e)

    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.model.save(self.model_path)

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)