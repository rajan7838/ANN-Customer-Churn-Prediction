# ANN Customer Churn Prediction

## What is this?
This project uses an Artificial Neural Network (ANN) to predict whether a bank customer will leave (churn) based on their profile. It includes:

- **Data preprocessing**: handle categories, scale numbers
- **Model training**: ANN with dropout & early stopping
- **Experiment tracking**: MLflow logs parameters, metrics, and the model
- **REST API**: FastAPI serves predictions
- **Containerization**: Docker makes it easy to run anywhere
- **CI/CD**: GitHub Actions automatically tests and builds the Docker image

## How to use it

### 1. Setup
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt