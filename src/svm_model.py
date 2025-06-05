from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel='linear', C=1.0, gamma='scale'):
    model = SVC(kernel=kernel, C=C, gamma=gamma) # type: ignore
    model.fit(X_train, y_train)
    return model
