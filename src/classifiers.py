import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_logistic_regression(
    X_train,
    y_train
):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_knn(X_train, y_train):

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def save_classifier(model, path):

    joblib.dump(
        model,
        path
    )