# 🏠 California Housing Price Prediction

This project uses the **California housing dataset** to predict **median house value** based on features such as location, demographics, and housing attributes. Two machine learning models are implemented and compared: **Linear Regression** and **K-Nearest Neighbors (KNN)**.

Inspired by exercises from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

---

## 📁 Dataset Overview
Each row represents data for a neighborhood block in California and includes:

| Feature               | Description                                         |
|-----------------------|-----------------------------------------------------|
| `longitude`           | Longitude coordinate of the block                   |
| `latitude`            | Latitude coordinate of the block                    |
| `housing_median_age` | Median age of the houses in the block              |
| `total_rooms`         | Total number of rooms                              |
| `total_bedrooms`      | Total number of bedrooms                           |
| `population`          | Total population of the block                      |
| `households`          | Number of households                               |
| `median_income`       | Median income of the block (in tens of thousands)  |
| `median_house_value`  | Target variable – Median house value               |
| `ocean_proximity`     | Categorical feature – How close the area is to ocean |

---

## 🧠 Models Used

- **Linear Regression**
- **K-Nearest Neighbors Regressor (KNN)**

Both models are evaluated and compared based on their prediction accuracy.

---

## 🛠️ Tech Stack
- Python  
- Jupyter Notebook  
- `pandas`, `numpy` – Data wrangling  
- `matplotlib`, `seaborn` – Visualizations  
- `scikit-learn` – Machine learning algorithms  

