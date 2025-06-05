from src.load_data import load_data
from src.svm_model import train_svm
from src.evaluate import evaluate_model
from src.visualize import plot_decision_boundary

import numpy as np

def main():
    X_train, X_test, y_train, y_test = load_data()

    print("\n--- Linear Kernel SVM ---")
    linear_svm = train_svm(X_train, y_train, kernel='linear', C=1.0)
    evaluate_model(linear_svm, X_test, y_test)

    print("\n--- RBF Kernel SVM ---")
    rbf_svm = train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')
    evaluate_model(rbf_svm, X_test, y_test)

    # Optional visualization (reduced dimension)
    print("\nVisualizing decision boundary (linear kernel in 2D PCA)...")
    plot_decision_boundary(linear_svm, np.vstack((X_train, X_test)), 
                           np.hstack((y_train, y_test)), 
                           "Linear Kernel SVM (PCA Reduced)")

if __name__ == "__main__":
    main()
