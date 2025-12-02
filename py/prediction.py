import joblib

def predict(data):
    clf = joblib.load("C:/Users/User/Downloads/Iris.csv")
    return clf.predict(data)