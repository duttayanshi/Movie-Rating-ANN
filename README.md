# 🎬 Movie Rating Prediction Using Artificial Neural Network

## 📌 Project Overview

Movie Rating Prediction is a Machine Learning and Deep Learning project that predicts the rating of a movie based on different movie-related features.

The project uses an Artificial Neural Network (ANN) regression model trained on movie data.

## 🎯 Objective

The main objective of this project is to predict a movie's rating using:

- Genre
- Release Year
- Runtime
- Number of Votes
- Revenue
- Metascore

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## 🧠 Machine Learning Model

An Artificial Neural Network was developed using TensorFlow/Keras.

### Model Architecture

```text
Input Layer
     ↓
Dense Layer - 64 neurons (ReLU)
     ↓
Dense Layer - 32 neurons (ReLU)
     ↓
Dense Layer - 16 neurons (ReLU)
     ↓
Dropout - 20%
     ↓
Output Layer - 1 neuron (Linear)
## 📊 Model Performance

The model was evaluated using the test dataset.

| Metric | Score |
|---|---:|
| Mean Squared Error (MSE) | 0.533 |
| Mean Absolute Error (MAE) | 0.535 |
| R² Score | 0.444 |

The MAE indicates that the model's predictions differ from the actual movie ratings by approximately **0.535 rating points on average** on the test data.

## 🔄 Project Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Genre Encoding
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
ANN Model Training
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Streamlit Web Application
      ↓
Movie Rating Prediction

📁 Project Structure
Movie-Rating-ANN/
│
├── dataset/
│   └── movies.csv
│
├── model/
│   ├── movie_rating_ann.keras
│   ├── scaler.pkl
│   └── genre_encoder.pkl
│
├── notebooks/
│   └── movie_rating_ann.ipynb
│
├── app.py
├── requirements.txt
└── README.md

💻 How to Run the Project
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Open the Project Folder
cd Movie-Rating-ANN

💻 How to Run the Project
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Open the Project Folder
cd Movie-Rating-ANN


🎬 Web Application

The Streamlit application allows users to enter movie details such as:

Genre
Release Year
Runtime
Votes
Revenue
Metascore

After clicking Predict Movie Rating, the trained ANN model generates a predicted rating.

## 🖥️ Application Screenshot

![Movie Rating Predictor](screenshots/movie-rating-app(1).png)
![Movie Rating Predictor](screenshots/movie-rating-app(2).png)

📈 Visualizations

The project includes:

Actual vs Predicted Movie Ratings
Training vs Validation Loss

These visualizations help understand the model's predictions and training performance.

🚀 Future Improvements
Improve model performance using hyperparameter tuning.
Add more movie features.
Experiment with different neural network architectures.
Add a movie recommendation feature.
Deploy the Streamlit application online.
👩‍💻 Author

Yanshi

MCA Student | Aspiring Software Developer


