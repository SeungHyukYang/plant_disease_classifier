# models/xgb_weighted.py
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# class_weight 방식으로 처리 (SMOTE 없이)
def train_xgb_weighted(X_train, y_train, X_test):
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=3.17,
        random_state=42,
        eval_metric=['logloss', 'aucpr'],
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred)
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred
