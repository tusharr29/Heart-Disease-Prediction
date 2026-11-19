# ❤️ Heart Disease Prediction Using Machine Learning

An interactive Machine Learning application that predicts the presence of heart disease using a Support Vector Machine (SVM) classifier. Built using Python, Scikit-learn, and Streamlit.

## 🌐 Live Application

**Try the application here:**

[❤️ Heart Disease Prediction – Streamlit App](https://heart-disease-prediction-svm.streamlit.app/)

**GitHub Repository:** [Heart Disease Prediction](https://github.com/tusharr29/Heart-Disease-Prediction)

## 📌 Project Overview

The objective of this project is to develop a machine learning classification model that predicts whether a person is likely to have heart disease based on their medical information and clinical measurements.

This project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, statistical feature selection, model comparison, cross-validation, hyperparameter optimization, model evaluation, and deployment using Streamlit.

## ✨ Features

- Interactive patient information form.
- Heart disease prediction using a trained SVM model.
- Exploratory Data Analysis (EDA).
- Categorical feature encoding and numerical feature scaling.
- Statistical feature selection using the Chi-square test.
- Training and comparison of multiple classification algorithms.
- Cross-validation for model evaluation.
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Interactive Streamlit web application.
- Professional white-themed user interface.

## 📊 Machine Learning Workflow

### 1. Data Preprocessing and Exploratory Data Analysis

Performed data exploration and preprocessing to prepare the dataset for machine learning.

Key activities:
- Explored the dataset and its feature distributions.
- Analysed relationships between input features and the target variable.
- Encoded categorical variables using One-Hot Encoding.
- Applied StandardScaler for feature scaling.
- Prepared training and testing datasets.

### 2. Feature Selection

Used the Chi-square statistical test to identify categorical features associated with the target variable, `HeartDisease`.

The test was performed using a significance level of 0.05.

**Key observations:**
- ST_Slope_Up showed the strongest association.
- ST_Slope_Flat and ExerciseAngina_Y also showed strong associations.
- Eight out of nine individually tested categorical features had p-values below 0.05.

### 3. Model Training and Comparison

Trained and evaluated five different classification algorithms to identify the best-performing initial model.

Models experimented with:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

Used cross-validation as part of model evaluation and compared model performance using Accuracy and F1-score.

#### Initial Model Comparison

| Algorithm | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | **86.96%** | **88.46%** |
| K-Nearest Neighbors | 86.09% | 87.69% |
| Naive Bayes | 86.09% | 87.50% |
| Decision Tree | 77.83% | 79.68% |
| Support Vector Machine | 84.78% | 86.59% |

**Observation:**

Logistic Regression achieved the best initial performance with 86.96% accuracy and an F1-score of 88.46%.

After comparing the models, I decided to further experiment with SVM through hyperparameter optimization.

### 4. Hyperparameter Optimization

Focused on optimizing the Support Vector Machine classifier using two hyperparameter tuning techniques:

#### A. GridSearchCV

GridSearchCV was used to systematically evaluate specified combinations of hyperparameters using cross-validation.

Parameters explored:

- C: 1, 10, 20, 30
- Kernel: Linear, RBF

**Results:**

| C | Kernel | Mean CV Score |
|---:|---|---:|
| 30 | Linear | **84.52%** |
| 1 | Linear | 84.41% |
| 10 | Linear | 84.41% |
| 20 | Linear | 84.41% |
| 1 | RBF | 82.56% |
| 10 | RBF | 80.82% |
| 20 | RBF | 79.74% |
| 30 | RBF | 79.19% |

**Best configuration:**
- C = 30
- Kernel = Linear
- Mean Cross-Validation Score = 84.52%

#### B. RandomizedSearchCV

RandomizedSearchCV was used to explore randomly selected hyperparameter combinations and identify promising configurations.

Parameters explored:

- C: 1, 10, 20, 30, 40
- Kernel: Linear, RBF

**Results:**

| C | Kernel | Mean CV Score |
|---:|---|---:|
| 30 | Linear | **84.52%** |
| 40 | Linear | **84.52%** |
| 1 | Linear | 84.41% |
| 10 | Linear | 84.41% |
| 20 | Linear | 84.41% |
| 1 | RBF | 82.56% |
| 10 | RBF | 80.82% |
| 20 | RBF | 79.74% |
| 30 | RBF | 79.19% |
| 40 | RBF | 78.32% |

**Best configurations:**
- C = 30, Kernel = Linear
- C = 40, Kernel = Linear
- Mean Cross-Validation Score = 84.52%

#### Hyperparameter Optimization Comparison

| Technique | Best C | Best Kernel | Mean CV Score |
|---|---:|---|---:|
| GridSearchCV | 30 | Linear | 84.52% |
| RandomizedSearchCV | 30 / 40 | Linear | 84.52% |

**Observation:**

Both optimization techniques identified the Linear kernel as the best-performing option among the tested configurations.

### 5. Final SVM Model Evaluation

Selected the Linear SVM configuration with C = 30 for the deployed application.

The final model was evaluated on the test dataset.

#### Performance Metrics

| Evaluation Metric | Score |
|---|---:|
| Cross-Validation Score | 84.52% |
| Test Accuracy | **84.78%** |
| Precision (Class 1) | 88% |
| Recall (Class 1) | 86% |
| F1-Score (Class 1) | 87% |

#### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 82 | 16 |
| Actual 1 | 19 | 113 |

The model correctly classified 195 out of 230 test observations.

### 6. Streamlit Application

Built an interactive web application using Streamlit to make the trained SVM model accessible through a user-friendly interface.

**Application features:**
- Patient information input fields.
- Interactive dropdowns for categorical variables.
- Heart disease prediction.
- Prediction result display.
- Clean and professional white-themed interface.

The trained model, scaler, and feature column names are saved using Joblib and loaded into the Streamlit application for predictions.

**Live Demo:** [Launch Heart Disease Prediction App](https://heart-disease-prediction-svm.streamlit.app/)

## 🛠️ Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── main.ipynb
├── heart_disease_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/tusharr29/Heart-Disease-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

- Exploratory Data Analysis (EDA).
- Data preprocessing and feature encoding.
- Statistical feature selection using Chi-square.
- Classification algorithms and model comparison.
- Cross-validation.
- Hyperparameter optimization using GridSearchCV and RandomizedSearchCV.
- Support Vector Machines.
- Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Model serialization using Joblib.
- Building interactive machine learning applications using Streamlit.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes only. Its predictions are not medical diagnoses and should not be used for clinical decision-making. Always consult a qualified healthcare professional for medical advice.

## 👨‍💻 Author

**Tushar Deshmukh**

- GitHub: [tusharr29](https://github.com/tusharr29)
- Project Repository: [Heart Disease Prediction](https://github.com/tusharr29/Heart-Disease-Prediction)
- Live Application: [Streamlit Demo](https://heart-disease-prediction-svm.streamlit.app/)
