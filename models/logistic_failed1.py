from sklearn.linear_model import LogisticRegression

def train_logistic_failed1(X_train, y_train, X_test):
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred