import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/breast-cancer.csv")

    # Drop unwanted columns if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Encode target
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Malignant:1, Benign:0

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
