from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_sampling_strategies(X_train, y_train, X_test, y_test, strategies=[0.5, 0.7, 1.0]):
    """
    Compare multiple SMOTE sampling strategies and print performance metrics
    """
    f1_scores = []
    recall_scores = []

    for strategy in strategies:
        print(f"\n--- sampling_strategy={strategy} ---")
        smote = SMOTE(sampling_strategy=strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)

        f1_scores.append(f1)
        recall_scores.append(recall)

        print(classification_report(y_test, y_pred, digits=4))
        print(f"F1-score (class 1): {f1:.4f}")
        print(f"Recall    (class 1): {recall:.4f}")

    # 시각화
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=strategies, y=f1_scores, marker='o', label='F1-score')
    sns.lineplot(x=strategies, y=recall_scores, marker='o', label='Recall')
    plt.title('SMOTE Sampling Strategy vs Performance')
    plt.xlabel('Sampling Strategy')
    plt.ylabel('Score')
    plt.xticks(strategies)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def check_overfitting(X_train, y_train, X_test, y_test, sampling_strategy=1.0):
    """
    Compare train vs test accuracy to detect overfitting
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    train_score = model.score(X_resampled, y_resampled)
    test_score = model.score(X_test, y_test)

    print("\n--- Overfitting Check ---")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy:  {test_score:.4f}")
