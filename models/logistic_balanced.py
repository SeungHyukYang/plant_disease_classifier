from sklearn.linear_model import LogisticRegression

def train_logistic_balanced(X_train, y_train, X_test):
    model = LogisticRegression(
        solver='lbfgs', 
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred