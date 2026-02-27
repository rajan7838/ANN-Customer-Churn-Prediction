 ANN Customer Churn Prediction
A simple end-to-end machine learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN).

🚀 Live Demo
The app is deployed on AWS EC2. You can access it here:

FastAPI (API) – http://16.170.212.133:8000

API Documentation (Swagger) – http://16.170.212.133:8000/docs

Streamlit Dashboard – http://16.170.212.133:8501

📌 What It Does
Takes customer details (credit score, geography, age, balance, etc.)

Uses a trained ANN model to predict churn probability

Provides both a REST API (FastAPI) and an interactive UI (Streamlit)

🛠️ Tech Stack
Python 3.11

TensorFlow / Keras – model training

FastAPI – REST API

Streamlit – interactive dashboard

Docker – containerization

AWS EC2 – cloud deployment

📁 Project Structure
text
.
├── src/                      # helper modules
├── data/                     # dataset (customer_churn.csv)
├── models/                   # trained model (ignored by git)
├── artifacts/                # preprocessor (ignored by git)
├── .gitignore
├── Dockerfile
├── requirements.txt
├── train.py                  # training script
├── app.py                    # FastAPI app
├── streamlit_app.py          # Streamlit dashboard
└── README.md
🧪 Run Locally
1. Clone the repository
bash
git clone https://github.com/rajan7838/ANN-Customer-Churn-Prediction.git
cd ANN-Customer-Churn-Prediction
2. Install dependencies (use virtual environment)
bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# or venv\Scripts\activate    # Windows
pip install -r requirements.txt
3. Train the model
bash
python train.py
This creates models/ann_model.h5 and artifacts/preprocessor.pkl.

4. Run FastAPI server
bash
uvicorn app:app --reload
Open http://localhost:8000/docs

5. Run Streamlit dashboard
bash
streamlit run streamlit_app.py
Open http://localhost:8501

Note: Make sure FastAPI is running before using the Streamlit app.

🐳 Run with Docker
Build the image
bash
docker build -t ann-churn .
Create a network for communication
bash
docker network create churn-network
Run FastAPI container
bash
docker run -d -p 8000:8000 --network churn-network --name fastapi-app ann-churn
Run Streamlit container
bash
docker run -d -p 8501:8501 --network churn-network --name streamlit-app ann-churn streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Now access:

API: http://localhost:8000

Streamlit: http://localhost:8501

☁️ Deploy on AWS EC2 (as done in this project)
Launch an EC2 instance (Ubuntu, t3.micro, 30GB storage).

SSH into the instance.

Install Docker:

bash
sudo apt update && sudo apt install docker.io -y
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
Clone the repository and build the image.

Run containers using the same Docker commands above.

Open ports 8000 and 8501 in the security group.

Your app is now live on your EC2 public IP!

📊 Dataset
The model is trained on the Bank Customer Churn dataset from Kaggle.
It contains 10,000 records with features like credit score, geography, age, balance, etc. The target is Exited (1 = churn, 0 = not churn).

🤝 Contributing
Feel free to fork, raise issues, or submit pull requests. All contributions are welcome!

📄 License
This project is open source under the MIT License.

👤 Author
Rajan – GitHub
