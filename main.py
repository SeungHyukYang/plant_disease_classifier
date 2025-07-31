from utils import load_data, split_data, print_score_summary

# from models.logistic_model import train_logistic
# from models.logistic_balanced import train_logistic_balanced
# from models.rf_model import train_random_forest
# from models.logistic_failed1 import train_logistic_failed1
# from models.rf_with_smote import train_rf_with_smote
# from models.smote_experiments import evaluate_sampling_strategies, check_overfitting
from models.xgb_weighted import train_xgb_weighted

def main():
    print("main.py starts")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 성능 실험
    evaluate_sampling_strategies(X_train, y_train, X_test, y_test, strategies=[0.5, 0.8, 1.0])

    # 과적합 확인
    check_overfitting(X_train, y_train, X_test, y_test, sampling_strategy=1.0)

if __name__ == "__main__":
    main()