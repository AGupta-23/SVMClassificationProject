# 🧠 SVM Classifier - Heart Disease Prediction

This project implements a Support Vector Machine (SVM) classifier to predict the presence of heart disease using the popular `heart.csv` dataset. It includes modular Python files for data loading, preprocessing, training, evaluation, and visualization.

---

## 📊 Dataset

We use the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) which contains key health metrics like age, sex, cholesterol, resting blood pressure, etc., to predict a binary target: `0` (no disease) or `1` (disease).

---

## 🧰 Project Structure

svm_classifier_project/
│
├── data_loader.py # Load and explore dataset
├── preprocess.py # Clean and prepare data
├── train_svm.py # Train the SVM classifier
├── evaluate_model.py # Evaluate using metrics
├── visualize.py # Plot confusion matrix, ROC curve, etc.
├── main.py # Main execution script
├── heart.csv # Dataset file
└── README.md # Project overview

yaml
Copy
Edit

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/svm-classifier.git
   cd svm-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the project:

bash
Copy
Edit
python main.py
🧪 Features
Cleaned and normalized dataset

Train/test split with stratification

SVM classifier with hyperparameter tuning

Accuracy, precision, recall, F1-score, ROC-AUC

Visualizations: Confusion Matrix, ROC Curve

📉 Evaluation Metrics
Metric	Value
Accuracy	~XX%
Precision	~XX%
Recall	~XX%
F1-Score	~XX%
ROC AUC	~XX

(Replace XX% with actual model results after training.)

📌 TODO
Add GUI or web interface (e.g., Streamlit)

Compare with other classifiers (Random Forest, KNN)

Deploy model for real-time inference
