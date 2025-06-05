# 🧠 SVM Classifier for Heart Disease Prediction

This project uses a **Support Vector Machine (SVM)** classifier to predict heart disease based on clinical features. The goal is to demonstrate the complete ML pipeline: data loading, preprocessing, model training, evaluation, and visualization.

---

## 📂 Project Structure

svm_classifier_project/
├── data_loader.py # Load and explore dataset
├── preprocess.py # Clean and prepare data
├── train_svm.py # Train the SVM model
├── evaluate_model.py # Evaluate model performance
├── visualize.py # Visualize results
├── main.py # Main script to run the pipeline
├── heart.csv # Dataset file
└── README.md # Project documentation

yaml
Copy
Edit

---

## 📊 Dataset

- **Source:** [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Description:** The dataset includes demographic and clinical attributes of patients and whether or not they had a heart disease event.

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/svm-classifier.git
cd svm-classifier
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Classifier
bash
Copy
Edit
python main.py
📈 Model Evaluation
The model is evaluated using the following metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC Score

Visualizations include:

✅ Confusion Matrix

📉 ROC Curve

🔍 Classification Report

🧪 Sample Output (Replace with actual)
Metric	Value
Accuracy	87.3%
Precision	84.6%
Recall	88.1%
F1-Score	86.3%
ROC AUC	90.2%

🛠️ Tools and Libraries
Python 3.x

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

📌 Future Enhancements
🔁 Compare with other ML models (Random Forest, Logistic Regression)

🌐 Deploy using Streamlit or Flask

🧠 Add hyperparameter tuning and cross-validation
