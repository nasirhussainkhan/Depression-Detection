import nltk
nltk.download('punkt')

from TweetModel import TweetClassifier, process_message
from math import log
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class DepressionDetection:


    def __init__(self):
        print("🔹 Loading dataset...")
        self.tweets = pd.read_csv('dataset/tweets.csv')
        self.tweets.drop(['Unnamed: 0'], axis=1, inplace=True)

        # Train-test split (98/2 split)
        trainIndex, testIndex = list(), list()
        for i in range(self.tweets.shape[0]):
            if np.random.uniform(0, 1) < 0.98:
                trainIndex.append(i)
            else:
                testIndex.append(i)

        self.trainData = self.tweets.iloc[trainIndex]
        self.testData = self.tweets.iloc[testIndex]

    def metrics(self, labels, predictions, model_name):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
            true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
            false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
            false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print(f"\n📊 Results for {model_name} model:")
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F-score:", round(fscore, 4))
        print("Accuracy:", round(accuracy * 100, 2), "%")

        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()


if __name__ == "__main__":

    obj = DepressionDetection()

    # TF-IDF Model
    sc_tf_idf = TweetClassifier(obj.trainData, 'tf-idf')
    preds_tf_idf = sc_tf_idf.predict(obj.testData['message'], 'tf-idf')
    obj.metrics(obj.testData['label'].reset_index(drop=True), list(preds_tf_idf.values()), "TF-IDF")

    # BOW Model
    sc_bow = TweetClassifier(obj.trainData, 'bow')
    preds_bow = sc_bow.predict(obj.testData['message'], 'bow')
    obj.metrics(obj.testData['label'].reset_index(drop=True), list(preds_bow.values()), "Bag of Words")

    # Sample Predictions for Presentation
    print("\n🧪 Sample Predictions:\n")

    test_sentences = [
        "Extreme sadness, lack of energy, hopelessness",
        "Loving how me and my lovely partner is talking about what we want.",
        "Hi hello depression and anxiety are the worst"
    ]

    for sentence in test_sentences:
        pm = process_message(sentence)
        prediction_tfidf = sc_tf_idf.classify(pm, 'tf-idf')
        prediction_bow = sc_bow.classify(pm, 'bow')
        print(f"📝 '{sentence}'")
        print(f"TF-IDF Prediction: {'Depressed' if prediction_tfidf == 1 else 'Not Depressed'}")
        print(f"BOW Prediction: {'Depressed' if prediction_bow == 1 else 'Not Depressed'}\n")
