import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    df = pd.read_csv("data/plant_disease_dataset.csv")
    X = df[['temperature', 'humidity', 'rainfall', 'soil_pH']]
    y = df['disease_present']
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def print_score_summary(y_test, y_pred):
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))