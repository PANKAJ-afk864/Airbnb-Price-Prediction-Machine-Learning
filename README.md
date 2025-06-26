# Airbnb NYC Price Prediction

This project predicts Airbnb listing prices in New York City using machine learning. It covers the full end-to-end pipeline from data cleaning, model building, evaluation, to deploying a web app using **Streamlit**.

---

## Objective

The goal is to build a machine learning model that can **predict the price** of an Airbnb listing based on features like:
- Location (latitude/longitude)
- Room type
- Neighbourhood group
- Number of reviews
- Availability
- And more

---

## Dataset

The dataset used is: **AB_NYC_2019.csv**  
---

## Project Workflow

### 1. **Data Preprocessing**
- Removed irrelevant columns (`id`, `name`, `host_name`, etc.)
- Handled missing values in `reviews_per_month`
- Removed extreme outliers (`price > $1000`)
- One-hot encoded categorical variables: `neighbourhood_group`, `room_type`

### 2. **Model Building**
- Trained a `RandomForestRegressor` on a sample of 10,000 rows (to optimize memory)
- Saved the trained model using `joblib`
- Saved the list of feature columns separately

### 3. **Streamlit Web App**
- Developed a user-friendly UI using Streamlit
- Takes input like latitude, room type, borough, etc.
- Predicts nightly price using the trained model
- Displays prediction in real-time

---

##  Technologies Used

| Type | Tech Stack |
|------|------------|
| Language | Python |
| Data Manipulation | Pandas |
| Machine Learning | Scikit-learn, RandomForest |
| Web Framework | Streamlit |
| Model Saving | Joblib |
| Deployment | Streamlit Cloud (optional) |

---

##  How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/PANKAJ-afk864/Airbnb-Price-Prediction-Machine-Learning.git
cd airbnb-price-prediction

## You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8506
  Network URL: http://192.168.0.105:8506
