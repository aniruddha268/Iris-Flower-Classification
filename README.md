#  Iris Flower Classification using K-Nearest Neighbors (KNN)

## Project Description

This project is a Machine Learning classification model built using the **K-Nearest Neighbors (KNN)** algorithm and the **Iris Dataset** from Scikit-learn.

The model learns from flower measurements and predicts the species of an unseen Iris flower based on four input features.

This project was completed as **Project 2** of the **DecodeLabs AI Engineering Internship**, focusing on the fundamentals of supervised learning and classification.

---

## Features

- Load the Iris dataset
- Perform feature scaling using StandardScaler
- Split data into training and testing sets
- Train a K-Nearest Neighbors (KNN) classifier
- Predict flower species
- Evaluate model accuracy
- Visualize results using a Confusion Matrix

---

## Dataset Information

**Dataset:** Iris Dataset (Scikit-learn)

### Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/aniruddha268/iris-flower-classification-knn.git
```

### Navigate into the project

```bash
cd iris-flower-classification-knn
```

### Create a virtual environment (Optional but Recommended)

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

## 📈 Results

The trained model achieved:

- ✅ 100% Accuracy on the test dataset
- ✅ Correct classification of all test samples
- ✅ Confusion Matrix visualization