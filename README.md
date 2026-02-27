# ANN Customer Churn Prediction
#A simple project to predict whether a bank customer will leave (churn) using an Artificial Neural Network (ANN).

#What it does
Takes customer details (credit score, geography, age, balance, etc.)

Predicts the probability that the customer will churn

Features
Data preprocessing & model training with TensorFlow/Keras

Experiment tracking with MLflow

REST API using FastAPI for predictions

Interactive dashboard with Streamlit

Docker containerization

CI/CD pipeline with GitHub Actions

How to use it locally
Clone the repository

bash
git clone https://github.com/rajan7838/ANN-Customer-Churn-Prediction.git
cd ANN-Customer-Churn-Prediction
Install dependencies

bash
pip install -r requirements.txt
Train the model

bash
python train.py
This will create the model file and log experiments with MLflow.

Run the FastAPI app

bash
uvicorn app:app --reload
Open http://localhost:8000/docs to test the API.

Run the Streamlit dashboard

bash
streamlit run streamlit_app.py
Using Docker
Build the image:

bash
docker build -t ann-churn .
Run the container:

bash
docker run -p 8000:8000 ann-churn
Then visit http://localhost:8000.

CI/CD
Every push to the main branch triggers GitHub Actions to:

Install dependencies

Run tests (if any)

Build and push Docker image

Project Structure
text
.
├── .github/workflows/   # CI/CD pipeline
├── data/                # dataset (customer_churn.csv)
├── src/                 # helper modules
├── .gitignore
├── Dockerfile
├── README.md
├── app.py               # FastAPI app
├── requirements.txt
├── streamlit_app.py     # Streamlit dashboard
└── train.py             # training script
Dataset
The model uses the Bank Customer Churn dataset from Kaggle.

Author
Rajan