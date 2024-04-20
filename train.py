import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data():
    return datasets.load_iris(return_X_y=True)


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    joblib.dump(model, filename)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train)
    save_model(model, 'uplift_model.pkl')
    print('Good')
