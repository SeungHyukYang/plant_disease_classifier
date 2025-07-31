from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_rf_with_smote(X_train, y_train, X_test):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return model, y_pred