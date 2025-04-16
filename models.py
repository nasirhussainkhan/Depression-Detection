import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


class Model:
    def __init__(self):
        self.name = ''
        path = 'dataset/depressionDataset.csv'
        df = pd.read_csv(path)
        df = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'class']]

        # Handle missing values
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        self.split_data(df)

    def split_data(self, df):
        x = df.iloc[:, :-1].values  # Features: q1 to q10
        y = df.iloc[:, -1].values  # Label: class
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.4, random_state=24)

    def svm_classifier(self):
        self.name = 'SVM Classifier'
        return SVC().fit(self.x_train, self.y_train)

    def decisionTree_classifier(self):
        self.name = 'Decision Tree Classifier'
        return DecisionTreeClassifier().fit(self.x_train, self.y_train)

    def randomforest_classifier(self):
        self.name = 'Random Forest Classifier'
        return RandomForestClassifier().fit(self.x_train, self.y_train)

    def naiveBayes_classifier(self):
        self.name = 'Naive Bayes Classifier'
        return GaussianNB().fit(self.x_train, self.y_train)

    def knn_classifier(self):
        self.name = 'KNN Classifier'
        return KNeighborsClassifier().fit(self.x_train, self.y_train)

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        print(f"{self.name} has accuracy of {accuracy * 100:.2f} %\n")

        # Optional: print detailed metrics (comment out if not needed)
        # print(classification_report(self.y_test, predictions))


if __name__ == '__main__':
    model = Model()

    classifiers = {
        "SVM": model.svm_classifier(),
        "Decision Tree": model.decisionTree_classifier(),
        "Random Forest": model.randomforest_classifier(),
        "Naive Bayes": model.naiveBayes_classifier(),
        "KNN": model.knn_classifier()
    }

    print("\n🔍 Accuracy Results for All Models:\n")
    for name, clf in classifiers.items():
        model.name = name
        model.accuracy(clf)

